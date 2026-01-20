"""
Fill Models for Backtesting.

Provides different levels of realism for simulating order fills:
- Simple: Immediate fill at best price
- Queue: Models queue position and time priority
- Impact: Models market impact from large orders

Choose based on your needs:
- Quick iteration: Simple model
- Realistic limit orders: Queue model
- Large order analysis: Impact model
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import math
import random

from src.data_feed.schemas import Fill, KalshiOrderbook, KalshiTrade, Order, Side


@dataclass
class FillResult:
    """Result of fill simulation."""
    filled: bool
    quantity: int
    price: float
    slippage: float
    fill: Optional[Fill] = None


class BaseFillModel(ABC):
    """Base class for fill models."""

    @abstractmethod
    def try_fill(
        self,
        order: Order,
        orderbook: KalshiOrderbook,
        timestamp: datetime,
    ) -> FillResult:
        """Attempt to fill an order given current orderbook."""
        pass

    @abstractmethod
    def on_trade(self, trade: KalshiTrade):
        """Process a market trade (for queue-based models)."""
        pass


class SimpleFillModel(BaseFillModel):
    """
    Simple fill model - immediate fills at the best available price.

    Assumptions:
    - Orders fill immediately if price is marketable
    - Fill price is the best available price on contra side
    - No queue modeling
    - Simple slippage based on order size

    Best for: Quick backtests, liquid markets, small orders
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        partial_fill_prob: float = 0.0,
    ):
        """
        Initialize simple fill model.

        Args:
            slippage_bps: Base slippage in basis points
            partial_fill_prob: Probability of partial fill (0-1)
        """
        self.slippage_bps = slippage_bps
        self.partial_fill_prob = partial_fill_prob

    def try_fill(
        self,
        order: Order,
        orderbook: KalshiOrderbook,
        timestamp: datetime,
    ) -> FillResult:
        """
        Try to fill order at current prices.

        Fills if our limit price crosses the spread.
        """
        # Determine if we can fill
        if order.side == Side.BUY:
            if not orderbook.yes_asks:
                return FillResult(filled=False, quantity=0, price=0, slippage=0)
            best_price = orderbook.best_ask
            can_fill = order.price >= best_price
        else:
            if not orderbook.yes_bids:
                return FillResult(filled=False, quantity=0, price=0, slippage=0)
            best_price = orderbook.best_bid
            can_fill = order.price <= best_price

        if not can_fill:
            return FillResult(filled=False, quantity=0, price=0, slippage=0)

        # Calculate fill
        fill_qty = order.quantity

        # Random partial fill
        if self.partial_fill_prob > 0 and random.random() < self.partial_fill_prob:
            fill_qty = random.randint(1, order.quantity)

        # Calculate slippage
        slippage = self.slippage_bps / 100 / 100 * fill_qty
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

        return FillResult(
            filled=True,
            quantity=fill_qty,
            price=fill_price,
            slippage=abs(fill_price - best_price) / 100,
            fill=fill,
        )

    def on_trade(self, trade: KalshiTrade):
        """Simple model doesn't use trade flow."""
        pass


class QueueFillModel(BaseFillModel):
    """
    Queue-based fill model - simulates realistic order queue position.

    Models:
    - Time priority in the order queue
    - Fills happen as trades execute at your price level
    - Queue position affects fill probability

    Best for: Limit order strategies, market making backtests
    """

    def __init__(
        self,
        queue_position_pct: float = 0.5,
        aggressive_fill_prob: float = 0.8,
    ):
        """
        Initialize queue fill model.

        Args:
            queue_position_pct: Assumed position in queue (0=front, 1=back)
            aggressive_fill_prob: Fill probability for marketable orders
        """
        self.queue_position_pct = queue_position_pct
        self.aggressive_fill_prob = aggressive_fill_prob

        # Track pending orders: order_id -> (order, queue_ahead)
        self._pending: dict[str, tuple[Order, int]] = {}

    def place_order(self, order: Order, orderbook: KalshiOrderbook) -> int:
        """
        Place order in simulated queue.

        Returns estimated queue position.
        """
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

        queue_ahead = int(depth * self.queue_position_pct)
        self._pending[order.client_order_id] = (order, queue_ahead)

        return queue_ahead

    def try_fill(
        self,
        order: Order,
        orderbook: KalshiOrderbook,
        timestamp: datetime,
    ) -> FillResult:
        """
        Try to fill order - for aggressive orders.

        For passive orders, use place_order + on_trade.
        """
        # Check if order is aggressive (crosses spread)
        if order.side == Side.BUY:
            is_aggressive = orderbook.best_ask and order.price >= orderbook.best_ask
            fill_price = orderbook.best_ask
        else:
            is_aggressive = orderbook.best_bid and order.price <= orderbook.best_bid
            fill_price = orderbook.best_bid

        if is_aggressive and random.random() < self.aggressive_fill_prob:
            fill = Fill(
                order_id=order.client_order_id,
                market_ticker=order.market_ticker,
                side=order.side,
                price=fill_price,
                quantity=order.quantity,
                timestamp=timestamp,
                fee=order.quantity * 0.07,
            )

            return FillResult(
                filled=True,
                quantity=order.quantity,
                price=fill_price,
                slippage=abs(fill_price - order.price) / 100 if order.price else 0,
                fill=fill,
            )

        # For passive orders, place in queue
        if order.client_order_id not in self._pending:
            self.place_order(order, orderbook)

        return FillResult(filled=False, quantity=0, price=0, slippage=0)

    def on_trade(self, trade: KalshiTrade) -> list[Fill]:
        """
        Process market trade and check for fills.

        Returns list of fills generated by this trade.
        """
        fills = []

        for order_id, (order, queue_ahead) in list(self._pending.items()):
            # Check if trade could fill our order
            if order.market_ticker != trade.market_ticker:
                continue

            # Check price compatibility
            if order.side == Side.BUY:
                price_match = trade.price <= order.price
            else:
                price_match = trade.price >= order.price

            if not price_match:
                continue

            # Trade reduces queue ahead of us
            if queue_ahead > 0:
                queue_ahead -= trade.quantity
                self._pending[order_id] = (order, max(0, queue_ahead))

            # Check if we get filled
            if queue_ahead <= 0:
                fill_qty = min(
                    order.quantity - getattr(order, "filled_quantity", 0),
                    trade.quantity,
                )

                if fill_qty > 0:
                    fill = Fill(
                        order_id=order_id,
                        market_ticker=order.market_ticker,
                        side=order.side,
                        price=order.price,  # Fill at our limit price
                        quantity=fill_qty,
                        timestamp=trade.timestamp,
                        fee=fill_qty * 0.07,
                    )
                    fills.append(fill)

                    # Update order state
                    if not hasattr(order, "filled_quantity"):
                        order.filled_quantity = 0
                    order.filled_quantity += fill_qty

                    # Remove if fully filled
                    if order.filled_quantity >= order.quantity:
                        del self._pending[order_id]

        return fills

    def cancel_order(self, order_id: str):
        """Cancel a pending order."""
        self._pending.pop(order_id, None)

    def get_pending_orders(self) -> list[Order]:
        """Get list of pending orders."""
        return [order for order, _ in self._pending.values()]


class ImpactFillModel(BaseFillModel):
    """
    Market impact fill model - simulates price impact from large orders.

    Models:
    - Square-root market impact
    - Permanent and temporary impact
    - Impact on subsequent trades

    Best for: Large order backtests, capacity analysis
    """

    def __init__(
        self,
        daily_volume: int = 1000,
        volatility: float = 0.02,  # Daily volatility in price terms
        permanent_ratio: float = 0.5,
    ):
        """
        Initialize impact fill model.

        Args:
            daily_volume: Average daily volume in contracts
            volatility: Daily price volatility (as fraction)
            permanent_ratio: Fraction of impact that's permanent
        """
        self.daily_volume = daily_volume
        self.volatility = volatility
        self.permanent_ratio = permanent_ratio

        # Track cumulative impact per market
        self._cumulative_impact: dict[str, float] = {}

    def try_fill(
        self,
        order: Order,
        orderbook: KalshiOrderbook,
        timestamp: datetime,
    ) -> FillResult:
        """
        Fill order with market impact.

        Uses square-root impact model:
            impact = σ × √(Q / ADV)
        """
        market = order.market_ticker

        # Get base price
        if order.side == Side.BUY:
            if not orderbook.yes_asks:
                return FillResult(filled=False, quantity=0, price=0, slippage=0)
            base_price = orderbook.best_ask
            direction = 1
        else:
            if not orderbook.yes_bids:
                return FillResult(filled=False, quantity=0, price=0, slippage=0)
            base_price = orderbook.best_bid
            direction = -1

        # Calculate market impact
        participation = order.quantity / self.daily_volume if self.daily_volume > 0 else 0.1
        impact = self.volatility * math.sqrt(participation) * 100  # Impact in cents

        # Apply cumulative impact
        cumulative = self._cumulative_impact.get(market, 0)
        total_impact = impact + cumulative * direction

        # Calculate fill price
        fill_price = base_price + direction * total_impact

        # Update cumulative (permanent) impact
        permanent_impact = impact * self.permanent_ratio
        if market not in self._cumulative_impact:
            self._cumulative_impact[market] = 0
        self._cumulative_impact[market] += direction * permanent_impact

        fill = Fill(
            order_id=order.client_order_id,
            market_ticker=market,
            side=order.side,
            price=int(fill_price),
            quantity=order.quantity,
            timestamp=timestamp,
            fee=order.quantity * 0.07,
        )

        return FillResult(
            filled=True,
            quantity=order.quantity,
            price=fill_price,
            slippage=abs(fill_price - base_price) / 100,
            fill=fill,
        )

    def on_trade(self, trade: KalshiTrade):
        """Trade flow can decay temporary impact."""
        # Simple decay model - impact decays with each trade
        market = trade.market_ticker
        if market in self._cumulative_impact:
            decay = 0.99  # 1% decay per trade
            self._cumulative_impact[market] *= decay

    def get_cumulative_impact(self, market: str) -> float:
        """Get current cumulative impact for a market."""
        return self._cumulative_impact.get(market, 0)

    def reset_impact(self, market: Optional[str] = None):
        """Reset cumulative impact."""
        if market:
            self._cumulative_impact.pop(market, None)
        else:
            self._cumulative_impact.clear()


def create_fill_model(model_type: str, **kwargs) -> BaseFillModel:
    """
    Factory function to create fill models.

    Args:
        model_type: "simple", "queue", or "impact"
        **kwargs: Model-specific parameters

    Returns:
        Configured fill model
    """
    if model_type == "simple":
        return SimpleFillModel(
            slippage_bps=kwargs.get("slippage_bps", 5.0),
            partial_fill_prob=kwargs.get("partial_fill_prob", 0.0),
        )
    elif model_type == "queue":
        return QueueFillModel(
            queue_position_pct=kwargs.get("queue_position_pct", 0.5),
            aggressive_fill_prob=kwargs.get("aggressive_fill_prob", 0.8),
        )
    elif model_type == "impact":
        return ImpactFillModel(
            daily_volume=kwargs.get("daily_volume", 1000),
            volatility=kwargs.get("volatility", 0.02),
            permanent_ratio=kwargs.get("permanent_ratio", 0.5),
        )
    else:
        raise ValueError(f"Unknown fill model: {model_type}")
