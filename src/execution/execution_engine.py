"""
Execution Engine - Main coordinator for trade execution.

Orchestrates the full execution flow:
1. Liquidity check - Can we execute at acceptable cost?
2. Toxicity check - Is the flow too informed?
3. Risk limits - Within position/loss limits?
4. Strategy selection - TWAP, Iceberg, or direct?
5. Execution - Submit and track orders
6. Position tracking - Update positions on fills
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from src.data_feed.kalshi_client import KalshiRESTClient
from src.data_feed.schemas import (
    ArbitrageSignal,
    Fill,
    KalshiOrderbook,
    Order,
    Side,
    UnderlyingTick,
)
from src.risk import RiskManager, RiskAssessment

from .config import ExecutionConfig, TWAPConfig, IcebergConfig, HedgeConfig
from .liquidity import LiquidityAnalyzer, LiquidityCheck
from .order_manager import OrderManager
from .smart_orders import SmartOrderRouter, ExecutionResult
from .hedge_simulator import HedgeSimulator

logger = logging.getLogger(__name__)


@dataclass
class ExecutionDecision:
    """Decision made by execution engine."""
    timestamp: datetime
    signal: ArbitrageSignal

    # Decision
    execute: bool
    reason: str

    # Checks performed
    liquidity_check: Optional[LiquidityCheck]
    risk_assessment: Optional[RiskAssessment]

    # Execution plan
    strategy: str  # "limit", "twap", "iceberg"
    size: int
    price: int

    # Result (if executed)
    result: Optional[ExecutionResult] = None


@dataclass
class EngineState:
    """Current state of execution engine."""
    is_running: bool
    signals_received: int
    signals_executed: int
    signals_rejected: int
    total_volume: int
    total_pnl: float


class ExecutionEngine:
    """
    Main execution engine coordinating all trading operations.

    Flow for each signal:
    1. Check liquidity (can we execute?)
    2. Check toxicity and risk limits (should we execute?)
    3. Determine execution strategy (how to execute?)
    4. Execute and track fills
    5. Update positions and trigger hedging

    Usage:
        engine = ExecutionEngine(kalshi_client, risk_manager)
        engine.on_fill(lambda fill: print(f"Fill: {fill}"))

        await engine.start()

        # Process signals
        decision = await engine.execute_signal(signal, orderbook, underlying)

        await engine.stop()
    """

    def __init__(
        self,
        kalshi_client: KalshiRESTClient,
        risk_manager: RiskManager,
        config: Optional[ExecutionConfig] = None,
        hedge_config: Optional[HedgeConfig] = None,
    ):
        """
        Initialize execution engine.

        Args:
            kalshi_client: Kalshi REST client for order submission
            risk_manager: Risk manager for toxicity and limit checks
            config: Execution configuration
            hedge_config: Hedge configuration
        """
        self.config = config or ExecutionConfig()
        self.risk_manager = risk_manager

        # Components
        self.order_manager = OrderManager(
            kalshi_client,
            fill_timeout_seconds=self.config.fill_timeout_seconds,
            partial_fill_timeout_seconds=self.config.partial_fill_timeout_seconds,
        )

        self.liquidity = LiquidityAnalyzer(
            max_spread_cents=self.config.max_spread_cents,
            min_depth=self.config.min_depth,
            max_impact_pct=self.config.max_impact_pct,
            max_pct_of_depth=self.config.max_pct_of_depth,
        )

        self.router = SmartOrderRouter(
            self.order_manager,
            twap_config=TWAPConfig(),
            iceberg_config=IcebergConfig(),
        )

        self.hedger = HedgeSimulator(hedge_config)

        # Callbacks
        self._fill_callbacks: list[Callable[[Fill], None]] = []
        self._decision_callbacks: list[Callable[[ExecutionDecision], None]] = []

        # State
        self._running = False
        self._stats = {
            "signals_received": 0,
            "signals_executed": 0,
            "signals_rejected": 0,
            "total_volume": 0,
        }

        # Register for fills
        self.order_manager.on_fill(self._handle_fill)

    def on_fill(self, callback: Callable[[Fill], None]):
        """Register callback for fill events."""
        self._fill_callbacks.append(callback)

    def on_decision(self, callback: Callable[[ExecutionDecision], None]):
        """Register callback for execution decisions."""
        self._decision_callbacks.append(callback)

    async def start(self):
        """Start execution engine."""
        self._running = True
        await self.order_manager.start()
        logger.info("Execution engine started")

    async def stop(self):
        """Stop execution engine."""
        self._running = False
        await self.order_manager.stop()
        logger.info("Execution engine stopped")

    async def execute_signal(
        self,
        signal: ArbitrageSignal,
        orderbook: KalshiOrderbook,
        underlying: Optional[UnderlyingTick] = None,
    ) -> ExecutionDecision:
        """
        Evaluate and potentially execute a trading signal.

        Args:
            signal: Arbitrage signal from basis calculator
            orderbook: Current Kalshi orderbook
            underlying: Current ES futures price (for hedging)

        Returns:
            ExecutionDecision with execution details and result
        """
        self._stats["signals_received"] += 1

        # Update risk manager with latest data
        self.risk_manager.update_orderbook(orderbook)

        # 1. Check liquidity
        liquidity_check = self.liquidity.can_execute(
            orderbook,
            signal.side,
            signal.recommended_size,
        )

        if not liquidity_check.can_execute:
            return self._reject_signal(
                signal,
                f"Liquidity: {liquidity_check.reason}",
                liquidity_check=liquidity_check,
            )

        # 2. Check risk (toxicity + limits)
        risk_assessment = self.risk_manager.evaluate_signal(signal, orderbook)

        if not risk_assessment.approved:
            return self._reject_signal(
                signal,
                f"Risk: {risk_assessment.rejection_reason.value}",
                liquidity_check=liquidity_check,
                risk_assessment=risk_assessment,
            )

        # 3. Determine execution size and strategy
        size = min(
            risk_assessment.adjusted_size,
            liquidity_check.max_size_no_impact,
            self.config.max_pct_of_depth * liquidity_check.depth_at_touch,
        )
        size = int(size)

        if size < 1:
            return self._reject_signal(
                signal,
                "Size reduced to zero",
                liquidity_check=liquidity_check,
                risk_assessment=risk_assessment,
            )

        # Select strategy
        strategy = self._select_strategy(size, liquidity_check)

        # Determine price
        if signal.side == Side.BUY:
            if self.config.use_aggressive_pricing:
                price = orderbook.best_ask or signal.kalshi_mid
            else:
                price = orderbook.best_bid or (int(signal.kalshi_mid) - 1)
        else:
            if self.config.use_aggressive_pricing:
                price = orderbook.best_bid or signal.kalshi_mid
            else:
                price = orderbook.best_ask or (int(signal.kalshi_mid) + 1)

        price = int(price)

        # 4. Execute
        decision = ExecutionDecision(
            timestamp=datetime.utcnow(),
            signal=signal,
            execute=True,
            reason="OK",
            liquidity_check=liquidity_check,
            risk_assessment=risk_assessment,
            strategy=strategy,
            size=size,
            price=price,
        )

        try:
            result = await self.router.execute(
                market=signal.market_ticker,
                side=signal.side,
                quantity=size,
                orderbook=orderbook,
                strategy=strategy,
            )

            decision.result = result

            if result.success:
                self._stats["signals_executed"] += 1
                self._stats["total_volume"] += result.total_filled

                # Handle hedging
                if underlying and self.hedger.enabled:
                    # Calculate delta from signal
                    delta = signal.fair_probability  # Simplified delta
                    self.hedger.on_kalshi_fill(
                        Fill(
                            order_id="",
                            market_ticker=signal.market_ticker,
                            side=signal.side,
                            price=int(result.avg_fill_price),
                            quantity=result.total_filled,
                            timestamp=datetime.utcnow(),
                            fee=result.total_fees,
                        ),
                        delta=delta,
                        es_price=underlying.price,
                    )

                logger.info(
                    f"Executed {signal.side.value} {result.total_filled}@{result.avg_fill_price:.0f} "
                    f"via {strategy} on {signal.market_ticker}"
                )
            else:
                logger.warning(f"Execution failed: {result.error}")

        except Exception as e:
            logger.error(f"Execution error: {e}")
            decision.reason = f"Error: {e}"
            decision.execute = False

        # Notify callbacks
        self._notify_decision(decision)

        return decision

    def _select_strategy(self, size: int, liquidity: LiquidityCheck) -> str:
        """Select execution strategy based on size and liquidity."""
        if self.config.default_strategy != "limit":
            return self.config.default_strategy

        # Auto-select based on size vs liquidity
        if size <= 10:
            return "limit"
        elif size <= liquidity.depth_at_touch * 0.5:
            return "twap"
        else:
            return "iceberg"

    def _reject_signal(
        self,
        signal: ArbitrageSignal,
        reason: str,
        liquidity_check: Optional[LiquidityCheck] = None,
        risk_assessment: Optional[RiskAssessment] = None,
    ) -> ExecutionDecision:
        """Create rejection decision."""
        self._stats["signals_rejected"] += 1

        decision = ExecutionDecision(
            timestamp=datetime.utcnow(),
            signal=signal,
            execute=False,
            reason=reason,
            liquidity_check=liquidity_check,
            risk_assessment=risk_assessment,
            strategy="",
            size=0,
            price=0,
        )

        logger.debug(f"Signal rejected: {reason}")
        self._notify_decision(decision)

        return decision

    def _handle_fill(self, fill: Fill):
        """Handle fill from order manager."""
        # Update risk manager's position tracking
        self.risk_manager.record_fill(fill)

        # Notify callbacks
        for callback in self._fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

    def _notify_decision(self, decision: ExecutionDecision):
        """Notify decision callbacks."""
        for callback in self._decision_callbacks:
            try:
                callback(decision)
            except Exception as e:
                logger.error(f"Decision callback error: {e}")

    def update_es_price(self, tick: UnderlyingTick):
        """Update ES price for hedge tracking."""
        self.hedger.update_es_price(tick)
        self.hedger.check_rehedge(tick.price)

    def get_state(self) -> EngineState:
        """Get current engine state."""
        return EngineState(
            is_running=self._running,
            signals_received=self._stats["signals_received"],
            signals_executed=self._stats["signals_executed"],
            signals_rejected=self._stats["signals_rejected"],
            total_volume=self._stats["total_volume"],
            total_pnl=self.risk_manager.positions._current_equity - self.risk_manager.positions.initial_capital,
        )

    def get_hedge_position(self):
        """Get current hedge position."""
        return self.hedger.get_position()

    def get_delta_exposure(self):
        """Get current delta exposure."""
        return self.hedger.get_delta_exposure()
