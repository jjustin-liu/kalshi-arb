"""
Liquidity Analysis for Execution Decisions.

Analyzes orderbook state to determine if a market is liquid enough
to execute a trade without excessive market impact.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import math

from src.data_feed.schemas import KalshiOrderbook, Side


@dataclass
class LiquidityCheck:
    """Result of liquidity analysis."""
    can_execute: bool
    reason: str

    # Metrics
    spread: int  # Bid-ask spread in cents
    best_bid: Optional[int]
    best_ask: Optional[int]
    bid_depth: int  # Total bid depth
    ask_depth: int  # Total ask depth
    depth_at_touch: int  # Depth at best price for our side

    # Impact estimate
    estimated_impact: float  # Expected price impact (0-1)
    effective_price: float  # Expected fill price after impact

    # Size recommendations
    max_size_no_impact: int  # Max size with minimal impact
    max_size_available: int  # Max size available in book


class LiquidityAnalyzer:
    """
    Analyzes market liquidity to determine execution feasibility.

    Checks:
    - Bid-ask spread (tight spread = more liquid)
    - Depth at best price (more depth = less impact)
    - Total book depth
    - Estimated market impact

    Usage:
        analyzer = LiquidityAnalyzer(max_spread=5, min_depth=10)
        check = analyzer.can_execute(orderbook, Side.BUY, quantity=20)
        if check.can_execute:
            # Proceed with execution
        else:
            print(f"Cannot execute: {check.reason}")
    """

    def __init__(
        self,
        max_spread_cents: int = 5,
        min_depth: int = 10,
        max_impact_pct: float = 0.01,
        max_pct_of_depth: float = 0.50,
    ):
        """
        Initialize liquidity analyzer.

        Args:
            max_spread_cents: Maximum acceptable bid-ask spread
            min_depth: Minimum contracts at best price
            max_impact_pct: Maximum acceptable price impact
            max_pct_of_depth: Maximum percentage of top-level depth to take
        """
        self.max_spread_cents = max_spread_cents
        self.min_depth = min_depth
        self.max_impact_pct = max_impact_pct
        self.max_pct_of_depth = max_pct_of_depth

    def can_execute(
        self,
        orderbook: KalshiOrderbook,
        side: Side,
        quantity: int,
    ) -> LiquidityCheck:
        """
        Check if an order can be executed with acceptable liquidity.

        Args:
            orderbook: Current orderbook state
            side: Trade side (BUY or SELL)
            quantity: Desired trade size

        Returns:
            LiquidityCheck with feasibility and metrics
        """
        # Basic orderbook checks
        if not orderbook.yes_bids or not orderbook.yes_asks:
            return LiquidityCheck(
                can_execute=False,
                reason="Empty orderbook",
                spread=0,
                best_bid=None,
                best_ask=None,
                bid_depth=0,
                ask_depth=0,
                depth_at_touch=0,
                estimated_impact=1.0,
                effective_price=0.0,
                max_size_no_impact=0,
                max_size_available=0,
            )

        best_bid = orderbook.best_bid
        best_ask = orderbook.best_ask
        spread = orderbook.spread or 0

        # Calculate depths
        bid_depth = sum(lvl.quantity for lvl in orderbook.yes_bids)
        ask_depth = sum(lvl.quantity for lvl in orderbook.yes_asks)

        # Depth on our side (what we'll be hitting)
        if side == Side.BUY:
            depth_at_touch = orderbook.yes_asks[0].quantity if orderbook.yes_asks else 0
            available_depth = ask_depth
            levels = orderbook.yes_asks
        else:
            depth_at_touch = orderbook.yes_bids[0].quantity if orderbook.yes_bids else 0
            available_depth = bid_depth
            levels = orderbook.yes_bids

        # Check spread
        if spread > self.max_spread_cents:
            return LiquidityCheck(
                can_execute=False,
                reason=f"Spread too wide ({spread}c > {self.max_spread_cents}c)",
                spread=spread,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                depth_at_touch=depth_at_touch,
                estimated_impact=0.0,
                effective_price=0.0,
                max_size_no_impact=0,
                max_size_available=available_depth,
            )

        # Check minimum depth
        if depth_at_touch < self.min_depth:
            return LiquidityCheck(
                can_execute=False,
                reason=f"Insufficient depth ({depth_at_touch} < {self.min_depth})",
                spread=spread,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                depth_at_touch=depth_at_touch,
                estimated_impact=0.0,
                effective_price=0.0,
                max_size_no_impact=0,
                max_size_available=available_depth,
            )

        # Estimate market impact
        impact, effective_price = self._estimate_impact(
            levels, quantity, side
        )

        # Check if impact is acceptable
        if impact > self.max_impact_pct:
            return LiquidityCheck(
                can_execute=False,
                reason=f"Impact too high ({impact:.1%} > {self.max_impact_pct:.1%})",
                spread=spread,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                depth_at_touch=depth_at_touch,
                estimated_impact=impact,
                effective_price=effective_price,
                max_size_no_impact=int(depth_at_touch * 0.3),
                max_size_available=available_depth,
            )

        # Calculate max sizes
        max_size_no_impact = int(depth_at_touch * self.max_pct_of_depth)

        return LiquidityCheck(
            can_execute=True,
            reason="OK",
            spread=spread,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            depth_at_touch=depth_at_touch,
            estimated_impact=impact,
            effective_price=effective_price,
            max_size_no_impact=max_size_no_impact,
            max_size_available=available_depth,
        )

    def _estimate_impact(
        self,
        levels: list,
        quantity: int,
        side: Side,
    ) -> tuple[float, float]:
        """
        Estimate market impact for an order.

        Uses square-root impact model: impact ∝ √(quantity/depth)

        Returns:
            (impact_pct, effective_price) tuple
        """
        if not levels or quantity <= 0:
            return 0.0, 0.0

        # Calculate how much of the book we'd consume
        remaining = quantity
        total_cost = 0.0

        for level in levels:
            fill_at_level = min(remaining, level.quantity)
            total_cost += fill_at_level * level.price
            remaining -= fill_at_level
            if remaining <= 0:
                break

        if quantity == remaining:
            # Couldn't fill anything
            return 1.0, 0.0

        filled = quantity - remaining
        effective_price = total_cost / filled

        # Impact is difference from best price
        best_price = levels[0].price
        impact = abs(effective_price - best_price) / 100  # Convert to probability

        return impact, effective_price

    def get_recommended_size(
        self,
        orderbook: KalshiOrderbook,
        side: Side,
        target_size: int,
    ) -> int:
        """
        Get recommended order size based on liquidity.

        May reduce size from target if liquidity is insufficient.

        Args:
            orderbook: Current orderbook
            side: Trade side
            target_size: Desired size

        Returns:
            Recommended size (may be less than target)
        """
        check = self.can_execute(orderbook, side, target_size)

        if check.can_execute:
            # Limit to max % of top level depth
            return min(target_size, check.max_size_no_impact)

        # If we can't execute target size, try smaller
        if check.max_size_no_impact > 0:
            return check.max_size_no_impact

        return 0

    def get_effective_price(
        self,
        orderbook: KalshiOrderbook,
        side: Side,
        quantity: int,
    ) -> Optional[float]:
        """
        Get expected effective price for an order.

        Args:
            orderbook: Current orderbook
            side: Trade side
            quantity: Order size

        Returns:
            Expected fill price or None if unfillable
        """
        if side == Side.BUY:
            levels = orderbook.yes_asks
        else:
            levels = orderbook.yes_bids

        if not levels:
            return None

        _, effective = self._estimate_impact(levels, quantity, side)
        return effective if effective > 0 else None
