"""
Order Manager - Handles order lifecycle.

Manages order submission, tracking, cancellation, and fill handling
through the Kalshi REST API.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from src.data_feed.kalshi_client import KalshiRESTClient, KalshiAPIError
from src.data_feed.schemas import Fill, Order, OrderStatus, Side

logger = logging.getLogger(__name__)


@dataclass
class OrderState:
    """Extended order state with tracking info."""
    order: Order
    submitted_at: Optional[datetime] = None
    last_update: Optional[datetime] = None
    cancel_requested: bool = False
    error_message: Optional[str] = None


@dataclass
class OrderManagerStats:
    """Order manager statistics."""
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_partially_filled: int = 0
    orders_cancelled: int = 0
    orders_rejected: int = 0
    total_volume: int = 0
    total_fees: float = 0.0


class OrderManager:
    """
    Manages order lifecycle with the Kalshi API.

    Handles:
    - Order submission with unique client IDs
    - Order status tracking and updates
    - Partial fill handling
    - Timeout-based cancellation
    - Fill callbacks

    Usage:
        manager = OrderManager(client)
        manager.on_fill(lambda fill: print(f"Filled: {fill}"))

        order = Order(...)
        order_id = await manager.submit(order)

        # Later check status
        status = await manager.get_status(order_id)

        # Cancel if needed
        await manager.cancel(order_id)
    """

    def __init__(
        self,
        client: KalshiRESTClient,
        fill_timeout_seconds: float = 30.0,
        partial_fill_timeout_seconds: float = 60.0,
    ):
        """
        Initialize order manager.

        Args:
            client: Kalshi REST client
            fill_timeout_seconds: Cancel unfilled orders after this
            partial_fill_timeout_seconds: Cancel partial fills after this
        """
        self.client = client
        self.fill_timeout = fill_timeout_seconds
        self.partial_fill_timeout = partial_fill_timeout_seconds

        # Order tracking
        self._orders: dict[str, OrderState] = {}  # order_id -> OrderState
        self._client_id_map: dict[str, str] = {}  # client_order_id -> order_id

        # Callbacks
        self._fill_callbacks: list[Callable[[Fill], None]] = []
        self._order_callbacks: list[Callable[[Order], None]] = []

        # Statistics
        self.stats = OrderManagerStats()

        # Background task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    def on_fill(self, callback: Callable[[Fill], None]):
        """Register callback for fill events."""
        self._fill_callbacks.append(callback)

    def on_order_update(self, callback: Callable[[Order], None]):
        """Register callback for order status updates."""
        self._order_callbacks.append(callback)

    async def start(self):
        """Start background order monitoring."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_orders())
        logger.info("Order manager started")

    async def stop(self):
        """Stop order monitoring and cancel open orders."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Cancel any open orders
        await self.cancel_all()
        logger.info("Order manager stopped")

    async def submit(self, order: Order) -> str:
        """
        Submit an order to Kalshi.

        Args:
            order: Order to submit

        Returns:
            Exchange order ID

        Raises:
            KalshiAPIError: If submission fails
        """
        # Generate client order ID if not set
        if not order.client_order_id:
            order.client_order_id = str(uuid.uuid4())

        try:
            # Submit to exchange
            order_id = await self.client.create_order(order)

            # Update order state
            order.exchange_order_id = order_id
            order.status = OrderStatus.OPEN

            # Track order
            self._orders[order_id] = OrderState(
                order=order,
                submitted_at=datetime.utcnow(),
                last_update=datetime.utcnow(),
            )
            self._client_id_map[order.client_order_id] = order_id

            self.stats.orders_submitted += 1
            logger.info(f"Order submitted: {order_id} ({order.side.value} {order.quantity}@{order.price})")

            return order_id

        except KalshiAPIError as e:
            order.status = OrderStatus.REJECTED
            self.stats.orders_rejected += 1
            logger.error(f"Order rejected: {e}")
            raise

    async def cancel(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Exchange order ID

        Returns:
            True if cancel was successful
        """
        if order_id not in self._orders:
            logger.warning(f"Order not found: {order_id}")
            return False

        state = self._orders[order_id]
        state.cancel_requested = True

        success = await self.client.cancel_order(order_id)

        if success:
            state.order.status = OrderStatus.CANCELLED
            state.last_update = datetime.utcnow()
            self.stats.orders_cancelled += 1
            self._notify_order_update(state.order)
            logger.info(f"Order cancelled: {order_id}")

        return success

    async def cancel_all(self, market: Optional[str] = None):
        """
        Cancel all open orders.

        Args:
            market: Optional market ticker to filter by
        """
        for order_id, state in list(self._orders.items()):
            if state.order.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
                if market is None or state.order.market_ticker == market:
                    await self.cancel(order_id)

    async def get_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get current order status.

        Args:
            order_id: Exchange order ID

        Returns:
            OrderStatus or None if not found
        """
        if order_id not in self._orders:
            return None

        # Refresh from exchange
        try:
            data = await self.client.get_order(order_id)
            status = self._parse_order_status(data.get("status", ""))

            state = self._orders[order_id]
            old_status = state.order.status
            state.order.status = status
            state.last_update = datetime.utcnow()

            # Check for fills
            filled_qty = data.get("filled_count", 0)
            if filled_qty > state.order.filled_quantity:
                new_fills = filled_qty - state.order.filled_quantity
                avg_price = data.get("avg_price", state.order.price)

                # Create fill record
                fill = Fill(
                    order_id=order_id,
                    market_ticker=state.order.market_ticker,
                    side=state.order.side,
                    price=avg_price,
                    quantity=new_fills,
                    timestamp=datetime.utcnow(),
                    fee=new_fills * 0.07,  # 7 cents per contract
                )

                state.order.filled_quantity = filled_qty
                state.order.avg_fill_price = avg_price

                self._notify_fill(fill)

            if status != old_status:
                self._notify_order_update(state.order)

            return status

        except KalshiAPIError as e:
            logger.error(f"Failed to get order status: {e}")
            return None

    async def _monitor_orders(self):
        """Background task to monitor open orders and handle timeouts."""
        while self._running:
            try:
                now = datetime.utcnow()

                for order_id, state in list(self._orders.items()):
                    # Skip completed orders
                    if state.order.status in (
                        OrderStatus.FILLED,
                        OrderStatus.CANCELLED,
                        OrderStatus.REJECTED,
                    ):
                        continue

                    # Refresh status
                    await self.get_status(order_id)

                    # Check timeouts
                    if state.submitted_at:
                        age = (now - state.submitted_at).total_seconds()

                        # Full timeout for unfilled orders
                        if (
                            state.order.status == OrderStatus.OPEN and
                            state.order.filled_quantity == 0 and
                            age > self.fill_timeout
                        ):
                            logger.info(f"Order timeout (unfilled): {order_id}")
                            await self.cancel(order_id)

                        # Partial fill timeout
                        elif (
                            state.order.status == OrderStatus.PARTIALLY_FILLED and
                            age > self.partial_fill_timeout
                        ):
                            logger.info(f"Order timeout (partial): {order_id}")
                            await self.cancel(order_id)

                await asyncio.sleep(1.0)  # Check every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Order monitor error: {e}")
                await asyncio.sleep(5.0)

    def _parse_order_status(self, status_str: str) -> OrderStatus:
        """Parse order status string from API."""
        status_map = {
            "pending": OrderStatus.PENDING,
            "open": OrderStatus.OPEN,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }
        return status_map.get(status_str.lower(), OrderStatus.PENDING)

    def _notify_fill(self, fill: Fill):
        """Notify fill callbacks."""
        self.stats.total_volume += fill.quantity
        self.stats.total_fees += fill.fee

        if fill.quantity == self._orders.get(fill.order_id, OrderState(Order(
            "", Side.BUY, 0, 0, ""
        ))).order.quantity:
            self.stats.orders_filled += 1
        else:
            self.stats.orders_partially_filled += 1

        for callback in self._fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

    def _notify_order_update(self, order: Order):
        """Notify order update callbacks."""
        for callback in self._order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Order callback error: {e}")

    def get_open_orders(self, market: Optional[str] = None) -> list[Order]:
        """Get all open orders, optionally filtered by market."""
        orders = []
        for state in self._orders.values():
            if state.order.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
                if market is None or state.order.market_ticker == market:
                    orders.append(state.order)
        return orders

    def get_position_from_fills(self, market: str) -> int:
        """Calculate position from fills for a market."""
        position = 0
        for state in self._orders.values():
            if state.order.market_ticker == market:
                filled = state.order.filled_quantity
                if state.order.side == Side.BUY:
                    position += filled
                else:
                    position -= filled
        return position
