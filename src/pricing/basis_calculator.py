"""Basis calculator for arbitrage signal generation."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.data_feed.schemas import (
    ArbitrageSignal,
    KalshiMarket,
    KalshiOrderbook,
    Side,
    UnderlyingTick,
)

from .binary_pricer import BinaryOptionPricer
from .vol_surface import VolatilityManager

logger = logging.getLogger(__name__)


@dataclass
class FeeStructure:
    """Kalshi fee structure."""
    maker_fee_rate: float = 0.0  # Maker fees are 0
    taker_fee_rate: float = 0.07  # 7 cents per contract per side
    exchange_fee_rate: float = 0.0

    def calculate_fee(self, price: int, quantity: int, is_maker: bool) -> float:
        """Calculate total fee in dollars."""
        if is_maker:
            return quantity * self.maker_fee_rate
        else:
            return quantity * self.taker_fee_rate


@dataclass
class SlippageModel:
    """Simple slippage model based on order size and spread."""
    base_slippage_bps: float = 10  # 10 bps base slippage
    size_impact_bps: float = 1  # 1 bp per contract

    def estimate_slippage(
        self,
        quantity: int,
        spread: int,
        depth: int,
    ) -> float:
        """
        Estimate expected slippage in probability points (0-1).

        Args:
            quantity: Order size in contracts
            spread: Current bid-ask spread in cents
            depth: Available liquidity at best price

        Returns:
            Expected slippage as probability (0-1)
        """
        # Base slippage from spread
        spread_slippage = spread / 2 / 100  # Half spread in probability terms

        # Size-dependent slippage
        if depth > 0:
            fill_ratio = min(1.0, quantity / depth)
            size_slippage = fill_ratio * self.size_impact_bps / 100
        else:
            size_slippage = self.base_slippage_bps / 100

        return spread_slippage + size_slippage


class BasisCalculator:
    """
    Calculates basis between fair value and Kalshi implied probability.

    Basis = Fair Probability (from Black-Scholes) - Implied Probability (from Kalshi mid)

    Positive basis = Kalshi is cheap, buy YES
    Negative basis = Kalshi is rich, sell YES (buy NO)
    """

    def __init__(
        self,
        pricer: Optional[BinaryOptionPricer] = None,
        vol_manager: Optional[VolatilityManager] = None,
        fee_structure: Optional[FeeStructure] = None,
        slippage_model: Optional[SlippageModel] = None,
        min_edge_threshold: float = 0.01,  # 100 bps minimum edge
    ):
        self.pricer = pricer or BinaryOptionPricer()
        self.vol_manager = vol_manager or VolatilityManager()
        self.fee_structure = fee_structure or FeeStructure()
        self.slippage_model = slippage_model or SlippageModel()
        self.min_edge_threshold = min_edge_threshold

    def calculate_signal(
        self,
        market: KalshiMarket,
        orderbook: KalshiOrderbook,
        underlying: UnderlyingTick,
        toxicity_score: float = 0.0,
        max_position_size: int = 100,
    ) -> Optional[ArbitrageSignal]:
        """
        Calculate arbitrage signal for a market.

        Args:
            market: Kalshi market metadata
            orderbook: Current Kalshi orderbook
            underlying: Current underlying price
            toxicity_score: Current toxicity score (0-1)
            max_position_size: Maximum position size in contracts

        Returns:
            ArbitrageSignal if opportunity exists, None otherwise
        """
        # Validate inputs
        if market.strike_price is None:
            logger.debug(f"No strike price for market {market.ticker}")
            return None

        if orderbook.mid_price is None:
            logger.debug(f"No orderbook for market {market.ticker}")
            return None

        # Get volatility
        vol = self.vol_manager.get_vol(
            underlying.price,
            market.strike_price,
            market.time_to_expiry,
        )

        if vol is None:
            # Fall back to default volatility
            vol = 0.15  # 15% default
            logger.debug(f"Using default volatility {vol} for {market.ticker}")

        # Calculate fair probability
        price_result = self.pricer.price(
            spot=underlying.price,
            strike=market.strike_price,
            volatility=vol,
            time_to_expiry=market.time_to_expiry,
        )

        fair_prob = price_result.probability
        implied_prob = orderbook.implied_probability

        # Calculate basis
        basis = fair_prob - implied_prob

        # Determine trade direction
        side = Side.BUY if basis > 0 else Side.SELL

        # Calculate expected costs
        # Use taker fee assumption (conservative)
        expected_fees = self.fee_structure.taker_fee_rate / 100  # Per contract as probability

        # Estimate slippage
        depth = (
            orderbook.yes_asks[0].quantity if side == Side.BUY and orderbook.yes_asks
            else orderbook.yes_bids[0].quantity if side == Side.SELL and orderbook.yes_bids
            else 0
        )
        spread = orderbook.spread or 2

        expected_slippage = self.slippage_model.estimate_slippage(
            quantity=10,  # Estimate for small order
            spread=spread,
            depth=depth,
        )

        # Net edge after costs
        net_edge = abs(basis) - expected_fees - expected_slippage

        # Calculate recommended size based on edge and liquidity
        recommended_size = self._calculate_position_size(
            net_edge=net_edge,
            depth=depth,
            max_size=max_position_size,
        )

        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(
            net_edge=net_edge,
            vol_confidence=0.8,  # TODO: get from vol manager
            liquidity_depth=depth,
            time_to_expiry=market.time_to_expiry,
        )

        return ArbitrageSignal(
            timestamp=datetime.utcnow(),
            market_ticker=market.ticker,
            underlying_symbol=underlying.symbol,
            underlying_price=underlying.price,
            strike_price=market.strike_price,
            kalshi_mid=orderbook.mid_price,
            fair_probability=fair_prob,
            implied_probability=implied_prob,
            basis=basis,
            expected_fees=expected_fees,
            expected_slippage=expected_slippage,
            net_edge=net_edge,
            toxicity_score=toxicity_score,
            volatility=vol,
            time_to_expiry=market.time_to_expiry,
            side=side,
            recommended_size=recommended_size,
            confidence=confidence,
        )

    def _calculate_position_size(
        self,
        net_edge: float,
        depth: int,
        max_size: int,
    ) -> int:
        """Calculate recommended position size."""
        if net_edge <= self.min_edge_threshold:
            return 0

        # Size based on edge (more edge = more size)
        edge_multiplier = min(2.0, net_edge / self.min_edge_threshold)

        # Limit by available liquidity
        liquidity_limit = int(depth * 0.5)  # Don't take more than 50% of top level

        # Base size
        base_size = 10

        size = int(base_size * edge_multiplier)
        size = min(size, liquidity_limit, max_size)

        return max(0, size)

    def _calculate_confidence(
        self,
        net_edge: float,
        vol_confidence: float,
        liquidity_depth: int,
        time_to_expiry: float,
    ) -> float:
        """Calculate confidence score (0-1) for the signal."""
        # Edge contribution
        edge_score = min(1.0, net_edge / 0.05)  # Max at 5% edge

        # Volatility confidence
        vol_score = vol_confidence

        # Liquidity score
        liquidity_score = min(1.0, liquidity_depth / 50)

        # Time decay (less confident near expiry due to gamma)
        time_score = min(1.0, time_to_expiry * 365 / 1)  # Full confidence if > 1 day

        # Weighted combination
        confidence = (
            0.3 * edge_score +
            0.3 * vol_score +
            0.2 * liquidity_score +
            0.2 * time_score
        )

        return confidence

    def update_underlying(self, timestamp: datetime, price: float):
        """Update volatility manager with new underlying price."""
        self.vol_manager.update(timestamp, price)


class SignalAggregator:
    """
    Aggregates signals across multiple markets.

    Helps identify the best opportunities when multiple markets have edge.
    """

    def __init__(self, max_signals: int = 10):
        self.max_signals = max_signals
        self._signals: list[ArbitrageSignal] = []

    def add_signal(self, signal: ArbitrageSignal):
        """Add signal to aggregator."""
        if signal.is_tradeable:
            self._signals.append(signal)

            # Sort by net edge
            self._signals.sort(key=lambda s: s.net_edge, reverse=True)

            # Keep only top N
            self._signals = self._signals[:self.max_signals]

    def get_best_signals(self, n: int = 5) -> list[ArbitrageSignal]:
        """Get top N signals by edge."""
        return self._signals[:n]

    def clear(self):
        """Clear all signals."""
        self._signals.clear()

    @property
    def has_opportunities(self) -> bool:
        """Check if there are any tradeable opportunities."""
        return len(self._signals) > 0
