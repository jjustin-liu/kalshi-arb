"""
Smart Order Execution Strategies.

Implements advanced execution algorithms to minimize market impact:
- TWAP: Time-Weighted Average Price - splits order over time
- Iceberg: Shows only partial size, refreshes as filled

These help avoid:
- Information leakage (others seeing your full order)
- Market impact (moving price against yourself)
- Pattern detection by other algorithms
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.data_feed.schemas import Fill, KalshiOrderbook, Order, Side

from .config import TWAPConfig, IcebergConfig
from .order_manager import OrderManager

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of smart order execution."""
    success: bool
    total_filled: int
    total_quantity: int
    avg_fill_price: float
    total_fees: float
    slices_executed: int
    elapsed_seconds: float
    error: Optional[str] = None

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled."""
        return self.total_filled / self.total_quantity if self.total_quantity > 0 else 0.0


class TWAPExecutor:
    """
    TWAP (Time-Weighted Average Price) Execution.

    Splits a large order into smaller slices executed at regular intervals.
    Randomizes timing and size to avoid detection.

    Why use TWAP:
    - Large orders moved all at once cause market impact
    - Splitting over time gets better average price
    - Randomization hides your pattern from other traders

    Usage:
        executor = TWAPExecutor(order_manager, config)
        result = await executor.execute(market, Side.BUY, quantity=100, orderbook=orderbook)
    """

    def __init__(
        self,
        order_manager: OrderManager,
        config: Optional[TWAPConfig] = None,
    ):
        """
        Initialize TWAP executor.

        Args:
            order_manager: Order manager for submitting orders
            config: TWAP configuration
        """
        self.order_manager = order_manager
        self.config = config or TWAPConfig()

    async def execute(
        self,
        market: str,
        side: Side,
        quantity: int,
        orderbook: KalshiOrderbook,
        price_limit: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute order using TWAP algorithm.

        Args:
            market: Market ticker
            side: Trade side
            quantity: Total quantity to execute
            orderbook: Current orderbook (for pricing)
            price_limit: Maximum price for buys, minimum for sells

        Returns:
            ExecutionResult with fill details
        """
        start_time = datetime.utcnow()
        cfg = self.config

        # Calculate slice parameters
        base_slice_size = quantity // cfg.num_slices
        remainder = quantity % cfg.num_slices
        base_interval = cfg.duration_seconds / cfg.num_slices

        # Generate slice sizes (with optional randomization)
        slice_sizes = []
        for i in range(cfg.num_slices):
            size = base_slice_size + (1 if i < remainder else 0)

            if cfg.randomize_size and size > 1:
                jitter = int(size * cfg.size_jitter_pct)
                size = size + random.randint(-jitter, jitter)
                size = max(1, size)

            slice_sizes.append(size)

        # Normalize to ensure we trade exact quantity
        total = sum(slice_sizes)
        if total != quantity:
            diff = quantity - total
            slice_sizes[-1] += diff

        # Execute slices
        total_filled = 0
        total_cost = 0.0
        total_fees = 0.0
        slices_executed = 0
        fills: list[Fill] = []

        for i, slice_size in enumerate(slice_sizes):
            if slice_size <= 0:
                continue

            # Calculate interval (with optional jitter)
            if i > 0:
                interval = base_interval
                if cfg.randomize_timing:
                    jitter = interval * cfg.timing_jitter_pct
                    interval = interval + random.uniform(-jitter, jitter)
                    interval = max(0.1, interval)

                await asyncio.sleep(interval)

            # Determine price
            # Use aggressive (cross spread) for final slice if configured
            is_final = (i == cfg.num_slices - 1)
            if is_final and cfg.aggressive_final_slice:
                # Cross the spread
                if side == Side.BUY:
                    price = orderbook.best_ask + 1 if orderbook.best_ask else 99
                else:
                    price = orderbook.best_bid - 1 if orderbook.best_bid else 1
            else:
                # Post at best bid/ask
                if side == Side.BUY:
                    price = orderbook.best_bid or (orderbook.best_ask - 1 if orderbook.best_ask else 50)
                else:
                    price = orderbook.best_ask or (orderbook.best_bid + 1 if orderbook.best_bid else 50)

            # Apply price limit
            if price_limit is not None:
                if side == Side.BUY:
                    price = min(price, price_limit)
                else:
                    price = max(price, price_limit)

            # Create and submit order
            order = Order(
                market_ticker=market,
                side=side,
                price=price,
                quantity=slice_size,
                client_order_id=f"twap_{market}_{i}_{datetime.utcnow().timestamp()}",
            )

            try:
                order_id = await self.order_manager.submit(order)

                # Wait for fill (with timeout)
                await asyncio.sleep(cfg.duration_seconds / cfg.num_slices / 2)

                # Check status
                status = await self.order_manager.get_status(order_id)

                if order.filled_quantity > 0:
                    total_filled += order.filled_quantity
                    total_cost += order.filled_quantity * (order.avg_fill_price or price)
                    total_fees += order.filled_quantity * 0.07
                    slices_executed += 1

                    logger.debug(f"TWAP slice {i+1}/{cfg.num_slices}: filled {order.filled_quantity}")

            except Exception as e:
                logger.error(f"TWAP slice {i+1} failed: {e}")

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return ExecutionResult(
            success=total_filled > 0,
            total_filled=total_filled,
            total_quantity=quantity,
            avg_fill_price=total_cost / total_filled if total_filled > 0 else 0.0,
            total_fees=total_fees,
            slices_executed=slices_executed,
            elapsed_seconds=elapsed,
        )


class IcebergExecutor:
    """
    Iceberg Order Execution.

    Shows only a small "visible" portion of the order. When that fills,
    automatically shows more. Hides true order size from the market.

    Why use Iceberg:
    - Other traders can't see your full order size
    - Avoids front-running by algorithms
    - Reduces information leakage

    Usage:
        executor = IcebergExecutor(order_manager, config)
        result = await executor.execute(market, Side.BUY, quantity=100, orderbook=orderbook)
    """

    def __init__(
        self,
        order_manager: OrderManager,
        config: Optional[IcebergConfig] = None,
    ):
        """
        Initialize Iceberg executor.

        Args:
            order_manager: Order manager for submitting orders
            config: Iceberg configuration
        """
        self.order_manager = order_manager
        self.config = config or IcebergConfig()

    async def execute(
        self,
        market: str,
        side: Side,
        quantity: int,
        orderbook: KalshiOrderbook,
        price: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute order using Iceberg algorithm.

        Args:
            market: Market ticker
            side: Trade side
            quantity: Total quantity to execute
            orderbook: Current orderbook (for pricing)
            price: Specific price to use (default: best bid/ask)

        Returns:
            ExecutionResult with fill details
        """
        start_time = datetime.utcnow()
        cfg = self.config

        remaining = quantity
        total_filled = 0
        total_cost = 0.0
        total_fees = 0.0
        slices_executed = 0

        # Determine initial price
        if price is None:
            if side == Side.BUY:
                price = orderbook.best_bid or (orderbook.best_ask - 1 if orderbook.best_ask else 50)
            else:
                price = orderbook.best_ask or (orderbook.best_bid + 1 if orderbook.best_bid else 50)

        current_price = price
        last_orderbook_mid = orderbook.mid_price or 50

        while remaining > 0:
            # Calculate visible size
            visible = min(remaining, cfg.visible_quantity)

            if cfg.randomize_visible and visible > cfg.min_visible:
                jitter = int(visible * cfg.visible_jitter_pct)
                visible = visible + random.randint(-jitter, jitter)
                visible = max(cfg.min_visible, min(visible, remaining))

            # Reprice if market moved
            if cfg.reprice_on_market_move and orderbook.mid_price:
                if abs(orderbook.mid_price - last_orderbook_mid) > 1:
                    if side == Side.BUY:
                        current_price = orderbook.best_bid or current_price
                    else:
                        current_price = orderbook.best_ask or current_price
                    current_price += cfg.price_adjustment
                    last_orderbook_mid = orderbook.mid_price

            # Create visible order
            order = Order(
                market_ticker=market,
                side=side,
                price=current_price,
                quantity=visible,
                client_order_id=f"iceberg_{market}_{slices_executed}_{datetime.utcnow().timestamp()}",
            )

            try:
                order_id = await self.order_manager.submit(order)

                # Wait for fill
                max_wait = 10.0  # Max wait per slice
                waited = 0.0
                while waited < max_wait:
                    await asyncio.sleep(0.5)
                    waited += 0.5

                    status = await self.order_manager.get_status(order_id)
                    if order.filled_quantity >= visible:
                        break

                # Process fill
                if order.filled_quantity > 0:
                    total_filled += order.filled_quantity
                    total_cost += order.filled_quantity * (order.avg_fill_price or current_price)
                    total_fees += order.filled_quantity * 0.07
                    remaining -= order.filled_quantity
                    slices_executed += 1

                    logger.debug(f"Iceberg slice filled: {order.filled_quantity}, remaining: {remaining}")

                    # Delay before showing more
                    if remaining > 0 and cfg.refresh_delay_seconds > 0:
                        delay = cfg.refresh_delay_seconds
                        if cfg.randomize_refresh:
                            delay = delay * random.uniform(0.5, 1.5)
                        await asyncio.sleep(delay)
                else:
                    # No fill - cancel and reprice
                    await self.order_manager.cancel(order_id)
                    break

            except Exception as e:
                logger.error(f"Iceberg slice failed: {e}")
                break

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return ExecutionResult(
            success=total_filled > 0,
            total_filled=total_filled,
            total_quantity=quantity,
            avg_fill_price=total_cost / total_filled if total_filled > 0 else 0.0,
            total_fees=total_fees,
            slices_executed=slices_executed,
            elapsed_seconds=elapsed,
        )


class SmartOrderRouter:
    """
    Routes orders to appropriate execution strategy.

    Automatically selects between:
    - Direct limit: Small orders, liquid markets
    - TWAP: Medium orders, need to minimize impact
    - Iceberg: Large orders, need to hide size
    """

    def __init__(
        self,
        order_manager: OrderManager,
        twap_config: Optional[TWAPConfig] = None,
        iceberg_config: Optional[IcebergConfig] = None,
    ):
        self.order_manager = order_manager
        self.twap = TWAPExecutor(order_manager, twap_config)
        self.iceberg = IcebergExecutor(order_manager, iceberg_config)

    async def execute(
        self,
        market: str,
        side: Side,
        quantity: int,
        orderbook: KalshiOrderbook,
        strategy: str = "auto",
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute order with appropriate strategy.

        Args:
            market: Market ticker
            side: Trade side
            quantity: Quantity to execute
            orderbook: Current orderbook
            strategy: "auto", "limit", "twap", or "iceberg"
            **kwargs: Additional arguments for specific strategy

        Returns:
            ExecutionResult
        """
        # Auto-select strategy based on size and liquidity
        if strategy == "auto":
            depth = (
                orderbook.yes_asks[0].quantity if side == Side.BUY and orderbook.yes_asks
                else orderbook.yes_bids[0].quantity if side == Side.SELL and orderbook.yes_bids
                else 0
            )

            if quantity <= 10 or quantity <= depth * 0.3:
                strategy = "limit"
            elif quantity <= depth:
                strategy = "twap"
            else:
                strategy = "iceberg"

        logger.info(f"Executing {quantity} {side.value} via {strategy}")

        if strategy == "limit":
            return await self._execute_limit(market, side, quantity, orderbook)
        elif strategy == "twap":
            return await self.twap.execute(market, side, quantity, orderbook, **kwargs)
        elif strategy == "iceberg":
            return await self.iceberg.execute(market, side, quantity, orderbook, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def _execute_limit(
        self,
        market: str,
        side: Side,
        quantity: int,
        orderbook: KalshiOrderbook,
    ) -> ExecutionResult:
        """Execute simple limit order."""
        start_time = datetime.utcnow()

        # Determine price
        if side == Side.BUY:
            price = orderbook.best_bid or (orderbook.best_ask - 1 if orderbook.best_ask else 50)
        else:
            price = orderbook.best_ask or (orderbook.best_bid + 1 if orderbook.best_bid else 50)

        order = Order(
            market_ticker=market,
            side=side,
            price=price,
            quantity=quantity,
            client_order_id=f"limit_{market}_{datetime.utcnow().timestamp()}",
        )

        try:
            order_id = await self.order_manager.submit(order)

            # Wait for fill
            await asyncio.sleep(5.0)
            await self.order_manager.get_status(order_id)

            elapsed = (datetime.utcnow() - start_time).total_seconds()

            return ExecutionResult(
                success=order.filled_quantity > 0,
                total_filled=order.filled_quantity,
                total_quantity=quantity,
                avg_fill_price=order.avg_fill_price or price,
                total_fees=order.filled_quantity * 0.07,
                slices_executed=1,
                elapsed_seconds=elapsed,
            )

        except Exception as e:
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            return ExecutionResult(
                success=False,
                total_filled=0,
                total_quantity=quantity,
                avg_fill_price=0.0,
                total_fees=0.0,
                slices_executed=0,
                elapsed_seconds=elapsed,
                error=str(e),
            )
