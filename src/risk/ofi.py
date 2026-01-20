"""
Order Flow Imbalance (OFI) Calculator.

OFI measures the net buying/selling pressure from orderbook changes.
It's calculated as the difference between bid and ask contributions:

    OFI = bid_contribution - ask_contribution

Where:
- bid_contribution = increase in bid depth - decrease in bid depth
- ask_contribution = increase in ask depth - decrease in ask depth

Positive OFI = net buying pressure (price likely to rise)
Negative OFI = net selling pressure (price likely to fall)

High absolute OFI values indicate informed trading activity.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from src.data_feed.schemas import KalshiOrderbook


@dataclass
class OFIUpdate:
    """Single OFI observation."""
    timestamp: datetime
    ofi: float  # Signed OFI value
    bid_contribution: float
    ask_contribution: float


@dataclass
class OFIState:
    """Current OFI state with statistics."""
    current_ofi: float
    ofi_zscore: float
    cumulative_ofi: float
    mean_ofi: float
    std_ofi: float
    num_observations: int
    last_update: datetime


class OFICalculator:
    """
    Calculates Order Flow Imbalance from orderbook updates.

    OFI captures the net order flow by tracking changes in orderbook depth.
    High positive OFI means buying pressure; high negative means selling pressure.

    Usage:
        ofi = OFICalculator(window_seconds=60)
        ofi.update(orderbook)
        state = ofi.get_state()
        print(f"OFI Z-score: {state.ofi_zscore}")
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        zscore_lookback: int = 100,
    ):
        """
        Initialize OFI calculator.

        Args:
            window_seconds: Rolling window for cumulative OFI
            zscore_lookback: Number of observations for z-score calculation
        """
        self.window_seconds = window_seconds
        self.zscore_lookback = zscore_lookback

        # Track previous orderbook state per market
        self._prev_orderbooks: dict[str, KalshiOrderbook] = {}

        # Rolling OFI observations per market
        self._ofi_history: dict[str, deque[OFIUpdate]] = {}

        # Statistics for z-score calculation
        self._ofi_values: dict[str, deque[float]] = {}

    def update(self, orderbook: KalshiOrderbook) -> Optional[float]:
        """
        Update OFI with new orderbook snapshot.

        Args:
            orderbook: New orderbook state

        Returns:
            Current OFI value (or None if first update for this market)
        """
        market = orderbook.market_ticker

        # Initialize tracking for new markets
        if market not in self._prev_orderbooks:
            self._prev_orderbooks[market] = orderbook
            self._ofi_history[market] = deque()
            self._ofi_values[market] = deque(maxlen=self.zscore_lookback)
            return None

        prev = self._prev_orderbooks[market]

        # Calculate bid contribution
        # Positive when bid depth increases or prices improve
        bid_contribution = self._calc_bid_contribution(prev, orderbook)

        # Calculate ask contribution
        # Positive when ask depth increases or prices worsen
        ask_contribution = self._calc_ask_contribution(prev, orderbook)

        # OFI = bid contribution - ask contribution
        # Positive OFI = net buying pressure
        ofi = bid_contribution - ask_contribution

        # Record update
        update = OFIUpdate(
            timestamp=orderbook.timestamp,
            ofi=ofi,
            bid_contribution=bid_contribution,
            ask_contribution=ask_contribution,
        )

        self._ofi_history[market].append(update)
        self._ofi_values[market].append(ofi)

        # Prune old observations
        self._prune_old(market, orderbook.timestamp)

        # Update previous orderbook
        self._prev_orderbooks[market] = orderbook

        return ofi

    def _calc_bid_contribution(
        self,
        prev: KalshiOrderbook,
        curr: KalshiOrderbook,
    ) -> float:
        """
        Calculate bid-side contribution to OFI.

        Contribution is positive when:
        - Bid depth increases at same price
        - Bid price improves (moves up)
        """
        contribution = 0.0

        prev_best_bid = prev.best_bid or 0
        curr_best_bid = curr.best_bid or 0

        # If bid price improved, count all depth as contribution
        if curr_best_bid > prev_best_bid:
            contribution += sum(lvl.quantity for lvl in curr.yes_bids)
        # If bid price worsened, negative contribution
        elif curr_best_bid < prev_best_bid:
            contribution -= sum(lvl.quantity for lvl in prev.yes_bids)
        else:
            # Same price - look at depth change
            prev_depth = prev.yes_bids[0].quantity if prev.yes_bids else 0
            curr_depth = curr.yes_bids[0].quantity if curr.yes_bids else 0
            contribution += curr_depth - prev_depth

        return contribution

    def _calc_ask_contribution(
        self,
        prev: KalshiOrderbook,
        curr: KalshiOrderbook,
    ) -> float:
        """
        Calculate ask-side contribution to OFI.

        Contribution is positive when:
        - Ask depth increases at same price
        - Ask price worsens (moves up)
        """
        contribution = 0.0

        prev_best_ask = prev.best_ask or 100
        curr_best_ask = curr.best_ask or 100

        # If ask price worsened (moved up), count all depth as contribution
        if curr_best_ask > prev_best_ask:
            contribution += sum(lvl.quantity for lvl in curr.yes_asks)
        # If ask price improved (moved down), negative contribution
        elif curr_best_ask < prev_best_ask:
            contribution -= sum(lvl.quantity for lvl in prev.yes_asks)
        else:
            # Same price - look at depth change
            prev_depth = prev.yes_asks[0].quantity if prev.yes_asks else 0
            curr_depth = curr.yes_asks[0].quantity if curr.yes_asks else 0
            contribution += curr_depth - prev_depth

        return contribution

    def _prune_old(self, market: str, current_time: datetime):
        """Remove observations outside the rolling window."""
        cutoff = current_time - timedelta(seconds=self.window_seconds)
        history = self._ofi_history[market]

        while history and history[0].timestamp < cutoff:
            history.popleft()

    def get_state(self, market: str) -> Optional[OFIState]:
        """
        Get current OFI state for a market.

        Returns:
            OFIState with current OFI, z-score, and statistics
        """
        if market not in self._ofi_history:
            return None

        history = self._ofi_history[market]
        values = self._ofi_values[market]

        if not history or len(values) < 2:
            return None

        # Current OFI
        current_ofi = history[-1].ofi

        # Cumulative OFI over window
        cumulative_ofi = sum(u.ofi for u in history)

        # Statistics for z-score
        ofi_array = np.array(list(values))
        mean_ofi = float(np.mean(ofi_array))
        std_ofi = float(np.std(ofi_array, ddof=1))

        # Z-score of cumulative OFI
        if std_ofi > 0:
            ofi_zscore = (current_ofi - mean_ofi) / std_ofi
        else:
            ofi_zscore = 0.0

        return OFIState(
            current_ofi=current_ofi,
            ofi_zscore=ofi_zscore,
            cumulative_ofi=cumulative_ofi,
            mean_ofi=mean_ofi,
            std_ofi=std_ofi,
            num_observations=len(values),
            last_update=history[-1].timestamp,
        )

    def get_toxicity_score(self, market: str) -> float:
        """
        Get toxicity score from OFI (0-1 scale).

        High absolute OFI z-score indicates informed trading.

        Returns:
            Toxicity score between 0 and 1
        """
        state = self.get_state(market)
        if state is None:
            return 0.0

        # Convert z-score to 0-1 scale using sigmoid-like function
        # |z| > 2 is concerning, |z| > 3 is very concerning
        abs_zscore = abs(state.ofi_zscore)

        # Score ramps up from 0 at z=0 to ~1 at z=4
        score = 1 - 1 / (1 + abs_zscore / 2)

        return min(1.0, score)

    def reset(self, market: Optional[str] = None):
        """Reset state for a market or all markets."""
        if market:
            self._prev_orderbooks.pop(market, None)
            self._ofi_history.pop(market, None)
            self._ofi_values.pop(market, None)
        else:
            self._prev_orderbooks.clear()
            self._ofi_history.clear()
            self._ofi_values.clear()
