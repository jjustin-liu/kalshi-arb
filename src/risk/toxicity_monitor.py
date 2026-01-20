"""
Combined Toxicity Monitor.

Aggregates multiple toxicity signals into a single score that indicates
how likely it is that informed traders are present in the market.

Components:
- OFI (25%): Order Flow Imbalance - directional pressure from orderbook
- VPIN (30%): Volume-synced informed trading probability
- Spread (20%): Widening spreads indicate uncertainty/toxicity
- Sweep (15%): Aggressive level clearing
- Imbalance (10%): Bid/ask depth imbalance

When combined toxicity > 0.6, trading should pause.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from src.data_feed.schemas import (
    KalshiOrderbook,
    KalshiTrade,
    ToxicityMetrics,
)

from .config import ToxicityConfig
from .ofi import OFICalculator
from .vpin import VPINCalculator
from .sweep_detector import SweepDetector


@dataclass
class SpreadState:
    """Spread monitoring state."""
    current_spread: float
    spread_zscore: float
    mean_spread: float
    std_spread: float
    num_observations: int


class ToxicityMonitor:
    """
    Monitors market toxicity by combining multiple indicators.

    Toxicity represents the probability that you're trading against
    informed counterparties who will move the market against you.

    Usage:
        monitor = ToxicityMonitor()
        monitor.update_orderbook(orderbook)
        monitor.update_trade(trade)
        metrics = monitor.get_metrics(market)
        if metrics.is_toxic:
            # Reduce exposure
    """

    def __init__(self, config: Optional[ToxicityConfig] = None):
        """
        Initialize toxicity monitor.

        Args:
            config: Toxicity configuration (uses defaults if None)
        """
        self.config = config or ToxicityConfig()

        # Component calculators
        self.ofi = OFICalculator(
            window_seconds=self.config.ofi_window_seconds,
        )
        self.vpin = VPINCalculator(
            bucket_size=self.config.vpin_bucket_size,
            num_buckets=self.config.vpin_num_buckets,
        )
        self.sweep = SweepDetector(
            time_window_ms=self.config.sweep_time_window_ms,
            min_levels=self.config.sweep_min_levels,
            cooldown_seconds=self.config.sweep_cooldown_seconds,
            imbalance_threshold=self.config.imbalance_ratio_threshold,
        )

        # Spread tracking per market
        self._spread_history: dict[str, deque[tuple[datetime, float]]] = {}
        self._spread_lookback = 100

    def update_orderbook(self, orderbook: KalshiOrderbook):
        """
        Update all components with new orderbook.

        Args:
            orderbook: New orderbook state
        """
        market = orderbook.market_ticker

        # Update OFI
        self.ofi.update(orderbook)

        # Update sweep detector
        self.sweep.update_orderbook(orderbook)

        # Update spread tracking
        if orderbook.spread is not None:
            self._update_spread(market, orderbook.timestamp, orderbook.spread)

    def update_trade(self, trade: KalshiTrade):
        """
        Update all components with new trade.

        Args:
            trade: Trade execution
        """
        # Update VPIN
        self.vpin.update(trade)

        # Update sweep detector
        self.sweep.update_trade(trade)

    def _update_spread(self, market: str, timestamp: datetime, spread: float):
        """Update spread tracking."""
        if market not in self._spread_history:
            self._spread_history[market] = deque(maxlen=self._spread_lookback)

        self._spread_history[market].append((timestamp, spread))

    def _get_spread_state(self, market: str) -> Optional[SpreadState]:
        """Get current spread state with z-score."""
        if market not in self._spread_history:
            return None

        history = self._spread_history[market]
        if len(history) < 10:
            return None

        spreads = np.array([s for _, s in history])
        current = spreads[-1]
        mean_spread = float(np.mean(spreads))
        std_spread = float(np.std(spreads, ddof=1))

        if std_spread > 0:
            zscore = (current - mean_spread) / std_spread
        else:
            zscore = 0.0

        return SpreadState(
            current_spread=current,
            spread_zscore=zscore,
            mean_spread=mean_spread,
            std_spread=std_spread,
            num_observations=len(history),
        )

    def _get_spread_toxicity(self, market: str) -> tuple[float, float]:
        """
        Get toxicity score from spread (0-1 scale).

        Returns:
            (toxicity_score, zscore) tuple
        """
        state = self._get_spread_state(market)
        if state is None:
            return 0.0, 0.0

        # Widening spread is concerning
        # Z-score > 2 is concerning, > 3 is very concerning
        zscore = max(0, state.spread_zscore)  # Only positive (widening) matters

        if zscore < 1:
            score = zscore * 0.2
        elif zscore < 2:
            score = 0.2 + (zscore - 1) * 0.3
        else:
            score = min(1.0, 0.5 + (zscore - 2) * 0.25)

        return score, state.spread_zscore

    def _get_imbalance_toxicity(self, orderbook: KalshiOrderbook) -> float:
        """
        Get toxicity score from bid/ask imbalance (0-1 scale).

        Args:
            orderbook: Current orderbook

        Returns:
            Toxicity score based on depth imbalance
        """
        is_imbalanced, ratio = self.sweep.check_imbalance(orderbook)

        if ratio == float('inf'):
            return 1.0

        # Normalize ratio to 0-1 scale
        # ratio of 1 = balanced (0 toxicity)
        # ratio of 3 = threshold (0.5 toxicity)
        # ratio > 5 = high toxicity
        threshold = self.config.imbalance_ratio_threshold

        if ratio <= 1.5:
            return 0.0
        elif ratio <= threshold:
            return 0.5 * (ratio - 1.5) / (threshold - 1.5)
        else:
            return min(1.0, 0.5 + 0.5 * (ratio - threshold) / threshold)

    def get_toxicity_score(
        self,
        market: str,
        orderbook: Optional[KalshiOrderbook] = None,
    ) -> float:
        """
        Get combined toxicity score for a market (0-1 scale).

        Args:
            market: Market ticker
            orderbook: Current orderbook (for imbalance check)

        Returns:
            Combined toxicity score between 0 and 1
        """
        cfg = self.config

        # OFI toxicity
        ofi_score = self.ofi.get_toxicity_score(market)

        # VPIN toxicity
        vpin_score = self.vpin.get_toxicity_score(market, cfg.vpin_threshold)

        # Spread toxicity
        spread_score, _ = self._get_spread_toxicity(market)

        # Sweep toxicity
        sweep_score = self.sweep.get_toxicity_score(market)

        # Imbalance toxicity
        if orderbook:
            imbalance_score = self._get_imbalance_toxicity(orderbook)
        else:
            imbalance_score = 0.0

        # Weighted combination
        combined = (
            cfg.weight_ofi * ofi_score +
            cfg.weight_vpin * vpin_score +
            cfg.weight_spread * spread_score +
            cfg.weight_sweep * sweep_score +
            cfg.weight_imbalance * imbalance_score
        )

        return min(1.0, combined)

    def should_pause_trading(
        self,
        market: str,
        orderbook: Optional[KalshiOrderbook] = None,
    ) -> bool:
        """
        Check if trading should pause due to toxicity.

        Args:
            market: Market ticker
            orderbook: Current orderbook

        Returns:
            True if toxicity is above pause threshold
        """
        score = self.get_toxicity_score(market, orderbook)
        return score > self.config.pause_threshold

    def get_metrics(
        self,
        market: str,
        orderbook: Optional[KalshiOrderbook] = None,
    ) -> ToxicityMetrics:
        """
        Get full toxicity metrics for a market.

        Populates the ToxicityMetrics dataclass from schemas.

        Args:
            market: Market ticker
            orderbook: Current orderbook

        Returns:
            ToxicityMetrics with all toxicity indicators
        """
        # Get OFI state
        ofi_state = self.ofi.get_state(market)
        ofi_value = ofi_state.current_ofi if ofi_state else 0.0
        ofi_zscore = ofi_state.ofi_zscore if ofi_state else 0.0

        # Get VPIN state
        vpin_state = self.vpin.get_state(market)
        vpin_value = vpin_state.vpin if vpin_state else 0.0

        # Get spread state
        spread_state = self._get_spread_state(market)
        spread_value = spread_state.current_spread if spread_state else 0.0
        spread_zscore = spread_state.spread_zscore if spread_state else 0.0

        # Get sweep state
        sweep_state = self.sweep.get_state(market)
        sweep_detected = sweep_state.sweep_active
        levels_cleared = max(
            sweep_state.bid_levels_cleared,
            sweep_state.ask_levels_cleared,
        )

        # Combined score
        toxicity_score = self.get_toxicity_score(market, orderbook)

        return ToxicityMetrics(
            timestamp=datetime.utcnow(),
            ofi=ofi_value,
            ofi_zscore=ofi_zscore,
            vpin=vpin_value,
            spread=spread_value,
            spread_zscore=spread_zscore,
            sweep_detected=sweep_detected,
            levels_cleared=levels_cleared,
            toxicity_score=toxicity_score,
        )

    def get_component_scores(
        self,
        market: str,
        orderbook: Optional[KalshiOrderbook] = None,
    ) -> dict[str, float]:
        """
        Get individual component toxicity scores.

        Useful for debugging and understanding what's driving toxicity.

        Returns:
            Dictionary with component names and their scores
        """
        cfg = self.config
        spread_score, _ = self._get_spread_toxicity(market)

        return {
            "ofi": self.ofi.get_toxicity_score(market),
            "vpin": self.vpin.get_toxicity_score(market, cfg.vpin_threshold),
            "spread": spread_score,
            "sweep": self.sweep.get_toxicity_score(market),
            "imbalance": (
                self._get_imbalance_toxicity(orderbook) if orderbook else 0.0
            ),
        }

    def reset(self, market: Optional[str] = None):
        """Reset state for a market or all markets."""
        self.ofi.reset(market)
        self.vpin.reset(market)
        self.sweep.reset(market)

        if market:
            self._spread_history.pop(market, None)
        else:
            self._spread_history.clear()
