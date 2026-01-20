"""
Fill Simulator for Backtesting.

Simulates realistic order fills for backtesting without actual execution.
Models:
- Queue position (where you are in the order queue)
- Partial fills
- Market impact
- Slippage
"""

import random
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.data_feed.schemas import Fill, KalshiOrderbook, KalshiTrade, Order, Side


@dataclass
class FillSimulationResult:
    """Result of fill simulation."""
    filled: bool
    fill_quantity: int
    fill_price: float
    slippage: float  # Difference from expected price
    market_impact: float  # Price movement caused
    queue_position: int  # Estimated position in queue
    fill: Optional[Fill] = None


class SimpleFillModel:
    """
    Simple fill model - immediate fill at best price.

    Assumes:
    - Orders fill immediately if liquidity available
    - No queue modeling
    - Simple slippage based on size vs depth

    Best for: Quick backtests, liquid markets
    """

    def __init__(
        self,
        base_slippage_bps: float = 5.0,
        fill_probability: float = 0.95,
    ):
        """
        Initialize simple fill model.

        Args:
            base_slippage_bps: Base slippage in basis points
            fill_probability: Probability that an order fills
        """
        self.base_slippage_bps = base_slippage_bps
        self.fill_probability = fill_probability

    def simulate_fill(
        self,
        order: Order,
        orderbook: KalshiOrderbook,
        timestamp: datetime,
    ) -> FillSimulationResult:
        """
        Simulate order fill.

        Args:
            order: Order to simulate
            orderbook: Current orderbook state
            timestamp: Simulation timestamp

        Returns:
            FillSimulationResult
        """
        # Check if we'd fill
        if random.random() > self.fill_probability:
            return FillSimulationResult(
                filled=False,
                fill_quantity=0,
                fill_price=0.0,
                slippage=0.0,
                market_impact=0.0,
                queue_position=0,
            )

        # Get available liquidity
        if order.side == Side.BUY:
            levels = orderbook.yes_asks
            best_price = orderbook.best_ask or order.price
        else:
            levels = orderbook.yes_bids
            best_price = orderbook.best_bid or order.price

        if not levels:
            return FillSimulationResult(
                filled=False,
                fill_quantity=0,
                fill_price=0.0,
                slippage=0.0,
                market_impact=0.0,
                queue_position=0,
            )

        # Calculate fill
        available = sum(lvl.quantity for lvl in levels if (
            (order.side == Side.BUY and lvl.price <= order.price) or
            (order.side == Side.SELL and lvl.price >= order.price)
        ))

        fill_qty = min(order.quantity, available)

        if fill_qty == 0:
            return FillSimulationResult(
                filled=False,
                fill_quantity=0,
                fill_price=0.0,
                slippage=0.0,
                market_impact=0.0,
                queue_position=0,
            )

        # Calculate slippage
        slippage_bps = self.base_slippage_bps * (1 + fill_qty / 100)
        slippage = slippage_bps / 100 / 100  # Convert to price units

        if order.side == Side.BUY:
            fill_price = best_price + slippage * 100
        else:
            fill_price = best_price - slippage * 100

        fill = Fill(
            order_id=order.client_order_id,
            market_ticker=order.market_ticker,
            side=order.side,
            price=int(fill_price),
            quantity=fill_qty,
            timestamp=timestamp,
            fee=fill_qty * 0.07,
        )

        return FillSimulationResult(
            filled=True,
            fill_quantity=fill_qty,
            fill_price=fill_price,
            slippage=slippage,
            market_impact=0.0,
            queue_position=0,
            fill=fill,
        )


class QueueFillModel:
    """
    Queue-based fill model - simulates realistic queue position.

    Models:
    - Queue position based on when order was placed
    - Fills happen as trades come through
    - Time priority matters

    Best for: Realistic backtests of limit order strategies
    """

    def __init__(
        self,
        queue_position_pct: float = 0.5,  # Assume middle of queue
        fill_rate_multiplier: float = 1.0,
    ):
        """
        Initialize queue fill model.

        Args:
            queue_position_pct: Assumed position in queue (0=front, 1=back)
            fill_rate_multiplier: Adjust fill aggressiveness
        """
        self.queue_position_pct = queue_position_pct
        self.fill_rate_multiplier = fill_rate_multiplier

        # Track pending orders per market
        self._pending_orders: dict[str, list[tuple[Order, int]]] = {}  # market -> [(order, queue_ahead)]

    def place_order(
        self,
        order: Order,
        orderbook: KalshiOrderbook,
    ) -> int:
        """
        Place order in simulated queue.

        Returns:
            Estimated queue position (contracts ahead of us)
        """
        market = order.market_ticker

        # Get depth at our price level
        if order.side == Side.BUY:
            depth = sum(
                lvl.quantity for lvl in orderbook.yes_bids
                if lvl.price == order.price
            )
        else:
            depth = sum(
                lvl.quantity for lvl in orderbook.yes_asks
                if lvl.price == order.price
            )

        # Assume we're at queue_position_pct of the depth
        queue_ahead = int(depth * self.queue_position_pct)

        if market not in self._pending_orders:
            self._pending_orders[market] = []

        self._pending_orders[market].append((order, queue_ahead))

        return queue_ahead

    def process_trade(
        self,
        trade: KalshiTrade,
        timestamp: datetime,
    ) -> list[Fill]:
        """
        Process a market trade and check for fills.

        Trades at our price level reduce our queue position.

        Returns:
            List of fills generated
        """
        market = trade.market_ticker
        fills = []

        if market not in self._pending_orders:
            return fills

        remaining_qty = trade.quantity
        new_pending = []

        for order, queue_ahead in self._pending_orders[market]:
            # Check if trade is at our price
            trade_matches = (
                (order.side == Side.BUY and trade.taker_side == Side.SELL and trade.price <= order.price) or
                (order.side == Side.SELL and trade.taker_side == Side.BUY and trade.price >= order.price)
            )

            if not trade_matches:
                new_pending.append((order, queue_ahead))
                continue

            # Trade reduces our queue
            if queue_ahead > 0:
                reduction = min(remaining_qty, queue_ahead)
                queue_ahead -= reduction
                remaining_qty -= reduction

            # Check if we get filled
            if queue_ahead == 0 and remaining_qty > 0:
                fill_qty = min(remaining_qty, order.quantity - order.filled_quantity)
                remaining_qty -= fill_qty

                order.filled_quantity += fill_qty

                fill = Fill(
                    order_id=order.client_order_id,
                    market_ticker=market,
                    side=order.side,
                    price=order.price,
                    quantity=fill_qty,
                    timestamp=timestamp,
                    fee=fill_qty * 0.07,
                )
                fills.append(fill)

            # Keep order if not fully filled
            if order.filled_quantity < order.quantity:
                new_pending.append((order, queue_ahead))

        self._pending_orders[market] = new_pending

        return fills

    def cancel_order(self, order: Order):
        """Cancel a pending order."""
        market = order.market_ticker
        if market in self._pending_orders:
            self._pending_orders[market] = [
                (o, q) for o, q in self._pending_orders[market]
                if o.client_order_id != order.client_order_id
            ]


class ImpactFillModel:
    """
    Market impact fill model - simulates price impact.

    Models:
    - Permanent and temporary impact
    - Square-root impact model
    - Impact decay over time

    Best for: Large order backtests, alpha decay analysis
    """

    def __init__(
        self,
        volatility: float = 0.15,  # Annualized volatility
        adv: int = 1000,  # Average daily volume
        permanent_impact_pct: float = 0.5,  # % of impact that's permanent
        impact_decay_seconds: float = 300.0,  # 5 minutes decay
    ):
        """
        Initialize impact fill model.

        Args:
            volatility: Annualized volatility for impact calculation
            adv: Average daily volume
            permanent_impact_pct: Fraction of impact that doesn't decay
            impact_decay_seconds: Time for temporary impact to decay
        """
        self.volatility = volatility
        self.adv = adv
        self.permanent_impact_pct = permanent_impact_pct
        self.impact_decay_seconds = impact_decay_seconds

        # Track cumulative impact
        self._cumulative_impact: dict[str, float] = {}

    def simulate_fill(
        self,
        order: Order,
        orderbook: KalshiOrderbook,
        timestamp: datetime,
    ) -> FillSimulationResult:
        """
        Simulate fill with market impact.

        Uses square-root impact model:
            impact = σ × √(Q / ADV) × sign(side)

        Args:
            order: Order to fill
            orderbook: Current orderbook
            timestamp: Simulation timestamp

        Returns:
            FillSimulationResult with impact estimates
        """
        market = order.market_ticker
        quantity = order.quantity

        # Calculate market impact using square-root model
        # impact = volatility * sqrt(quantity / ADV)
        participation_rate = quantity / self.adv if self.adv > 0 else 0.1
        impact = self.volatility * math.sqrt(participation_rate)

        # Daily vol to per-trade vol (rough approximation)
        impact = impact / math.sqrt(252 * 6.5 * 60)  # ~trades per day

        # Convert to price units (cents for Kalshi)
        impact_cents = impact * 100

        # Direction
        if order.side == Side.BUY:
            direction = 1
            base_price = orderbook.best_ask or order.price
        else:
            direction = -1
            base_price = orderbook.best_bid or order.price

        # Calculate fill price with impact
        fill_price = base_price + direction * impact_cents

        # Track cumulative impact
        permanent = impact_cents * self.permanent_impact_pct
        if market not in self._cumulative_impact:
            self._cumulative_impact[market] = 0.0
        self._cumulative_impact[market] += direction * permanent

        # Slippage is the difference from best price
        slippage = abs(fill_price - base_price) / 100

        fill = Fill(
            order_id=order.client_order_id,
            market_ticker=market,
            side=order.side,
            price=int(fill_price),
            quantity=quantity,
            timestamp=timestamp,
            fee=quantity * 0.07,
        )

        return FillSimulationResult(
            filled=True,
            fill_quantity=quantity,
            fill_price=fill_price,
            slippage=slippage,
            market_impact=impact_cents,
            queue_position=0,
            fill=fill,
        )

    def get_cumulative_impact(self, market: str) -> float:
        """Get cumulative permanent impact for a market."""
        return self._cumulative_impact.get(market, 0.0)

    def reset(self, market: Optional[str] = None):
        """Reset impact tracking."""
        if market:
            self._cumulative_impact.pop(market, None)
        else:
            self._cumulative_impact.clear()
