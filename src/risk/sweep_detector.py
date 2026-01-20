"""
Orderbook Sweep Detector.

Detects aggressive trading patterns where multiple price levels are
cleared in a short time window. Sweeps indicate:
- Urgent/informed trading (someone knows something)
- Large institutional orders
- Stop-loss cascades

When a sweep is detected, market makers should:
- Widen quotes or pause quoting
- Reduce size
- Wait for the market to stabilize
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from src.data_feed.schemas import KalshiOrderbook, KalshiTrade, Side


@dataclass
class SweepEvent:
    """Detected sweep event."""
    timestamp: datetime
    market_ticker: str
    side: Side  # BUY sweep or SELL sweep
    levels_cleared: int
    total_volume: int
    price_start: int
    price_end: int
    duration_ms: float


@dataclass
class SweepState:
    """Current sweep detection state."""
    sweep_active: bool
    last_sweep: Optional[SweepEvent]
    cooldown_until: Optional[datetime]
    bid_levels_cleared: int
    ask_levels_cleared: int
    recent_sweeps_count: int  # Sweeps in last minute


class SweepDetector:
    """
    Detects orderbook sweeps - when multiple price levels are cleared quickly.

    A sweep is identified when:
    1. Multiple price levels disappear in a short time window (< 500ms)
    2. Bid/ask ratio becomes extremely imbalanced (> 3x)
    3. Large trades execute through multiple price levels

    Usage:
        detector = SweepDetector(time_window_ms=500, min_levels=2)
        detector.update_orderbook(orderbook)
        detector.update_trade(trade)
        if detector.is_sweep_active(market):
            # Reduce exposure
    """

    def __init__(
        self,
        time_window_ms: int = 500,
        min_levels: int = 2,
        imbalance_threshold: float = 3.0,
        cooldown_seconds: float = 5.0,
    ):
        """
        Initialize sweep detector.

        Args:
            time_window_ms: Time window to detect sweep (milliseconds)
            min_levels: Minimum price levels cleared to count as sweep
            imbalance_threshold: Bid/ask ratio above which is considered imbalanced
            cooldown_seconds: Time to remain in "sweep mode" after detection
        """
        self.time_window_ms = time_window_ms
        self.min_levels = min_levels
        self.imbalance_threshold = imbalance_threshold
        self.cooldown_seconds = cooldown_seconds

        # Per-market state
        self._prev_orderbooks: dict[str, KalshiOrderbook] = {}
        self._orderbook_changes: dict[str, deque[tuple[datetime, int, int]]] = {}
        self._recent_trades: dict[str, deque[KalshiTrade]] = {}
        self._sweep_events: dict[str, deque[SweepEvent]] = {}
        self._cooldown_until: dict[str, Optional[datetime]] = {}

    def update_orderbook(self, orderbook: KalshiOrderbook) -> Optional[SweepEvent]:
        """
        Update with new orderbook and check for sweeps.

        Args:
            orderbook: New orderbook state

        Returns:
            SweepEvent if sweep detected, None otherwise
        """
        market = orderbook.market_ticker

        # Initialize for new markets
        if market not in self._prev_orderbooks:
            self._prev_orderbooks[market] = orderbook
            self._orderbook_changes[market] = deque(maxlen=100)
            self._sweep_events[market] = deque(maxlen=50)
            return None

        prev = self._prev_orderbooks[market]

        # Count levels cleared
        bid_levels_cleared = self._count_levels_cleared(
            prev.yes_bids, orderbook.yes_bids
        )
        ask_levels_cleared = self._count_levels_cleared(
            prev.yes_asks, orderbook.yes_asks
        )

        # Record change
        self._orderbook_changes[market].append(
            (orderbook.timestamp, bid_levels_cleared, ask_levels_cleared)
        )

        # Check for sweep
        sweep = self._check_for_sweep(market, orderbook.timestamp)

        # Update previous orderbook
        self._prev_orderbooks[market] = orderbook

        if sweep:
            self._record_sweep(sweep)

        return sweep

    def update_trade(self, trade: KalshiTrade):
        """
        Update with new trade (used to correlate with orderbook changes).

        Args:
            trade: Trade execution
        """
        market = trade.market_ticker

        if market not in self._recent_trades:
            self._recent_trades[market] = deque(maxlen=100)

        self._recent_trades[market].append(trade)

        # Prune old trades
        cutoff = trade.timestamp - timedelta(seconds=1)
        while (
            self._recent_trades[market] and
            self._recent_trades[market][0].timestamp < cutoff
        ):
            self._recent_trades[market].popleft()

    def _count_levels_cleared(
        self,
        prev_levels: list,
        curr_levels: list,
    ) -> int:
        """Count how many price levels were cleared."""
        if not prev_levels:
            return 0

        prev_prices = {lvl.price for lvl in prev_levels}
        curr_prices = {lvl.price for lvl in curr_levels}

        # Levels that existed before but don't exist now
        cleared = prev_prices - curr_prices
        return len(cleared)

    def _check_for_sweep(
        self,
        market: str,
        current_time: datetime,
    ) -> Optional[SweepEvent]:
        """Check if recent orderbook changes constitute a sweep."""
        changes = self._orderbook_changes[market]
        if not changes:
            return None

        # Look at changes within time window
        window_start = current_time - timedelta(milliseconds=self.time_window_ms)
        recent_changes = [
            (ts, bid, ask) for ts, bid, ask in changes
            if ts >= window_start
        ]

        if not recent_changes:
            return None

        # Sum up levels cleared
        total_bid_cleared = sum(bid for _, bid, _ in recent_changes)
        total_ask_cleared = sum(ask for _, _, ask in recent_changes)

        # Check if sweep threshold met
        if total_bid_cleared >= self.min_levels:
            # Bid sweep (selling pressure)
            return self._create_sweep_event(
                market, current_time, Side.SELL, total_bid_cleared
            )

        if total_ask_cleared >= self.min_levels:
            # Ask sweep (buying pressure)
            return self._create_sweep_event(
                market, current_time, Side.BUY, total_ask_cleared
            )

        return None

    def _create_sweep_event(
        self,
        market: str,
        timestamp: datetime,
        side: Side,
        levels_cleared: int,
    ) -> SweepEvent:
        """Create a sweep event."""
        # Get trade volume in recent window
        total_volume = 0
        price_start = 0
        price_end = 0

        if market in self._recent_trades:
            trades = self._recent_trades[market]
            matching_trades = [
                t for t in trades
                if t.taker_side == side
            ]
            if matching_trades:
                total_volume = sum(t.quantity for t in matching_trades)
                price_start = matching_trades[0].price
                price_end = matching_trades[-1].price

        return SweepEvent(
            timestamp=timestamp,
            market_ticker=market,
            side=side,
            levels_cleared=levels_cleared,
            total_volume=total_volume,
            price_start=price_start,
            price_end=price_end,
            duration_ms=self.time_window_ms,
        )

    def _record_sweep(self, sweep: SweepEvent):
        """Record sweep event and set cooldown."""
        market = sweep.market_ticker

        if market not in self._sweep_events:
            self._sweep_events[market] = deque(maxlen=50)

        self._sweep_events[market].append(sweep)
        self._cooldown_until[market] = (
            sweep.timestamp + timedelta(seconds=self.cooldown_seconds)
        )

    def is_sweep_active(self, market: str, current_time: Optional[datetime] = None) -> bool:
        """
        Check if a sweep is currently active (in cooldown period).

        Args:
            market: Market ticker
            current_time: Current timestamp (default: utcnow)

        Returns:
            True if in sweep cooldown, False otherwise
        """
        if market not in self._cooldown_until:
            return False

        cooldown = self._cooldown_until[market]
        if cooldown is None:
            return False

        now = current_time or datetime.utcnow()
        return now < cooldown

    def get_state(self, market: str) -> SweepState:
        """Get current sweep detection state for a market."""
        is_active = self.is_sweep_active(market)
        last_sweep = None
        cooldown = None

        if market in self._sweep_events and self._sweep_events[market]:
            last_sweep = self._sweep_events[market][-1]

        if market in self._cooldown_until:
            cooldown = self._cooldown_until[market]

        # Count recent sweeps (last minute)
        recent_count = 0
        if market in self._sweep_events:
            cutoff = datetime.utcnow() - timedelta(minutes=1)
            recent_count = sum(
                1 for s in self._sweep_events[market]
                if s.timestamp > cutoff
            )

        # Get current levels from most recent change
        bid_cleared = 0
        ask_cleared = 0
        if market in self._orderbook_changes and self._orderbook_changes[market]:
            _, bid_cleared, ask_cleared = self._orderbook_changes[market][-1]

        return SweepState(
            sweep_active=is_active,
            last_sweep=last_sweep,
            cooldown_until=cooldown,
            bid_levels_cleared=bid_cleared,
            ask_levels_cleared=ask_cleared,
            recent_sweeps_count=recent_count,
        )

    def get_toxicity_score(self, market: str) -> float:
        """
        Get toxicity score from sweep detection (0-1 scale).

        Returns:
            1.0 if sweep active, decaying based on recent sweeps
        """
        if self.is_sweep_active(market):
            return 1.0

        state = self.get_state(market)

        # Score based on recent sweep count
        if state.recent_sweeps_count >= 3:
            return 0.8
        elif state.recent_sweeps_count >= 2:
            return 0.5
        elif state.recent_sweeps_count >= 1:
            return 0.3
        else:
            return 0.0

    def check_imbalance(self, orderbook: KalshiOrderbook) -> tuple[bool, float]:
        """
        Check for extreme bid/ask imbalance.

        Args:
            orderbook: Current orderbook

        Returns:
            (is_imbalanced, imbalance_ratio) tuple
        """
        bid_depth = sum(lvl.quantity for lvl in orderbook.yes_bids)
        ask_depth = sum(lvl.quantity for lvl in orderbook.yes_asks)

        if bid_depth == 0 or ask_depth == 0:
            return True, float('inf')

        ratio = max(bid_depth / ask_depth, ask_depth / bid_depth)
        is_imbalanced = ratio > self.imbalance_threshold

        return is_imbalanced, ratio

    def reset(self, market: Optional[str] = None):
        """Reset state for a market or all markets."""
        if market:
            self._prev_orderbooks.pop(market, None)
            self._orderbook_changes.pop(market, None)
            self._recent_trades.pop(market, None)
            self._sweep_events.pop(market, None)
            self._cooldown_until.pop(market, None)
        else:
            self._prev_orderbooks.clear()
            self._orderbook_changes.clear()
            self._recent_trades.clear()
            self._sweep_events.clear()
            self._cooldown_until.clear()
