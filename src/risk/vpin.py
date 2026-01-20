"""
VPIN (Volume-Synchronized Probability of Informed Trading) Calculator.

VPIN measures the probability that trades are coming from informed traders
who have superior information about the asset's true value.

Key concept: Informed traders tend to trade on one side (buy if they know
price will go up, sell if they know price will go down), creating volume
imbalance.

VPIN = |sum(V_buy) - sum(V_sell)| / sum(V_total)

Where volumes are calculated over "buckets" of fixed volume (not time),
making it robust to varying trade frequency.

High VPIN (> 0.6) indicates toxic order flow - market makers should
widen spreads or reduce quote sizes.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.data_feed.schemas import KalshiTrade, Side


@dataclass
class VPINBucket:
    """A volume bucket for VPIN calculation."""
    buy_volume: int = 0
    sell_volume: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def total_volume(self) -> int:
        return self.buy_volume + self.sell_volume

    @property
    def imbalance(self) -> int:
        """Absolute volume imbalance."""
        return abs(self.buy_volume - self.sell_volume)

    @property
    def bucket_vpin(self) -> float:
        """VPIN for this single bucket."""
        if self.total_volume == 0:
            return 0.0
        return self.imbalance / self.total_volume


@dataclass
class VPINState:
    """Current VPIN state."""
    vpin: float  # 0-1 scale
    num_buckets: int
    total_buy_volume: int
    total_sell_volume: int
    total_volume: int
    last_update: datetime
    current_bucket_fill: float  # How full is current bucket (0-1)


class VPINCalculator:
    """
    Calculates VPIN from trade flow.

    VPIN uses volume-synchronized buckets (vs time-synchronized) to
    measure informed trading probability. Each bucket fills with a
    fixed amount of volume, then a new bucket starts.

    Usage:
        vpin = VPINCalculator(bucket_size=100, num_buckets=50)
        for trade in trades:
            vpin.update(trade)
        state = vpin.get_state()
        print(f"VPIN: {state.vpin:.2f}")
    """

    def __init__(
        self,
        bucket_size: int = 100,
        num_buckets: int = 50,
    ):
        """
        Initialize VPIN calculator.

        Args:
            bucket_size: Volume per bucket (contracts)
            num_buckets: Number of rolling buckets for VPIN calculation
        """
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets

        # Per-market state
        self._buckets: dict[str, deque[VPINBucket]] = {}
        self._current_bucket: dict[str, VPINBucket] = {}
        self._last_update: dict[str, datetime] = {}

    def update(self, trade: KalshiTrade) -> Optional[float]:
        """
        Update VPIN with new trade.

        Uses taker_side from trade to classify as buy or sell.

        Args:
            trade: Trade execution

        Returns:
            Current VPIN value (or None if insufficient data)
        """
        market = trade.market_ticker

        # Initialize for new markets
        if market not in self._buckets:
            self._buckets[market] = deque(maxlen=self.num_buckets)
            self._current_bucket[market] = VPINBucket(start_time=trade.timestamp)

        bucket = self._current_bucket[market]

        # Add volume to current bucket
        if trade.taker_side == Side.BUY:
            bucket.buy_volume += trade.quantity
        else:
            bucket.sell_volume += trade.quantity

        bucket.end_time = trade.timestamp
        self._last_update[market] = trade.timestamp

        # Check if bucket is full
        while bucket.total_volume >= self.bucket_size:
            # Calculate overflow
            overflow = bucket.total_volume - self.bucket_size

            # Finalize bucket (trim to exact size)
            if overflow > 0:
                # Proportionally remove overflow from buy/sell
                buy_ratio = bucket.buy_volume / bucket.total_volume
                overflow_buy = int(overflow * buy_ratio)
                overflow_sell = overflow - overflow_buy

                bucket.buy_volume -= overflow_buy
                bucket.sell_volume -= overflow_sell

            # Store completed bucket
            self._buckets[market].append(bucket)

            # Start new bucket with overflow
            new_bucket = VPINBucket(start_time=trade.timestamp)
            if overflow > 0:
                overflow_buy = int(overflow * buy_ratio)
                new_bucket.buy_volume = overflow_buy
                new_bucket.sell_volume = overflow - overflow_buy

            self._current_bucket[market] = new_bucket
            bucket = new_bucket

        return self.get_vpin(market)

    def get_vpin(self, market: str) -> Optional[float]:
        """
        Calculate current VPIN for a market.

        VPIN = |Σ(buy_volume) - Σ(sell_volume)| / Σ(total_volume)

        Returns:
            VPIN value (0-1) or None if insufficient data
        """
        if market not in self._buckets:
            return None

        buckets = self._buckets[market]
        if len(buckets) < 2:
            return None

        total_buy = sum(b.buy_volume for b in buckets)
        total_sell = sum(b.sell_volume for b in buckets)
        total_volume = total_buy + total_sell

        if total_volume == 0:
            return None

        vpin = abs(total_buy - total_sell) / total_volume
        return vpin

    def get_state(self, market: str) -> Optional[VPINState]:
        """
        Get detailed VPIN state for a market.

        Returns:
            VPINState with VPIN and related statistics
        """
        vpin = self.get_vpin(market)
        if vpin is None:
            return None

        buckets = self._buckets[market]
        current = self._current_bucket[market]

        total_buy = sum(b.buy_volume for b in buckets)
        total_sell = sum(b.sell_volume for b in buckets)

        return VPINState(
            vpin=vpin,
            num_buckets=len(buckets),
            total_buy_volume=total_buy,
            total_sell_volume=total_sell,
            total_volume=total_buy + total_sell,
            last_update=self._last_update.get(market, datetime.utcnow()),
            current_bucket_fill=current.total_volume / self.bucket_size,
        )

    def get_toxicity_score(self, market: str, threshold: float = 0.6) -> float:
        """
        Get toxicity score from VPIN (0-1 scale).

        VPIN naturally ranges from 0 to 1, but we normalize based on
        the threshold for concerning VPIN.

        Args:
            market: Market ticker
            threshold: VPIN threshold above which is concerning (default 0.6)

        Returns:
            Toxicity score between 0 and 1
        """
        vpin = self.get_vpin(market)
        if vpin is None:
            return 0.0

        # Below threshold/2 = low toxicity
        # At threshold = medium toxicity (0.5)
        # Above threshold = high toxicity
        if vpin <= threshold / 2:
            return vpin / threshold  # 0 to 0.5
        else:
            # Ramp up more steeply above threshold/2
            return min(1.0, 0.5 + (vpin - threshold / 2) / (threshold / 2))

    def get_trade_direction(self, market: str) -> Optional[Side]:
        """
        Get dominant trade direction from recent buckets.

        Returns:
            Side.BUY if more buying, Side.SELL if more selling, None if balanced
        """
        if market not in self._buckets:
            return None

        buckets = self._buckets[market]
        if not buckets:
            return None

        # Look at recent buckets
        recent = list(buckets)[-10:] if len(buckets) >= 10 else list(buckets)

        buy_vol = sum(b.buy_volume for b in recent)
        sell_vol = sum(b.sell_volume for b in recent)

        if buy_vol > sell_vol * 1.5:
            return Side.BUY
        elif sell_vol > buy_vol * 1.5:
            return Side.SELL
        else:
            return None

    def reset(self, market: Optional[str] = None):
        """Reset state for a market or all markets."""
        if market:
            self._buckets.pop(market, None)
            self._current_bucket.pop(market, None)
            self._last_update.pop(market, None)
        else:
            self._buckets.clear()
            self._current_bucket.clear()
            self._last_update.clear()
