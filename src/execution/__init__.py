"""
Execution Module.

Handles all aspects of order execution:
- Liquidity analysis (can we trade?)
- Order management (submit/track/cancel)
- Smart execution (TWAP, Iceberg)
- Fill simulation (for backtesting)
- Hedge tracking (theoretical ES position)

Usage:
    from src.execution import ExecutionEngine, ExecutionConfig

    engine = ExecutionEngine(
        kalshi_client=client,
        risk_manager=risk_mgr,
        config=ExecutionConfig(),
    )

    await engine.start()
    decision = await engine.execute_signal(signal, orderbook, underlying)
    await engine.stop()
"""

from .config import (
    ExecutionConfig,
    TWAPConfig,
    IcebergConfig,
    HedgeConfig,
)
from .liquidity import LiquidityAnalyzer, LiquidityCheck
from .order_manager import OrderManager, OrderManagerStats
from .smart_orders import (
    TWAPExecutor,
    IcebergExecutor,
    SmartOrderRouter,
    ExecutionResult,
)
from .fill_simulator import (
    SimpleFillModel,
    QueueFillModel,
    ImpactFillModel,
    FillSimulationResult,
)
from .hedge_simulator import (
    HedgeSimulator,
    HedgePosition,
    HedgeTrade,
    DeltaExposure,
)
from .execution_engine import (
    ExecutionEngine,
    ExecutionDecision,
    EngineState,
)

__all__ = [
    # Config
    "ExecutionConfig",
    "TWAPConfig",
    "IcebergConfig",
    "HedgeConfig",
    # Liquidity
    "LiquidityAnalyzer",
    "LiquidityCheck",
    # Order Management
    "OrderManager",
    "OrderManagerStats",
    # Smart Orders
    "TWAPExecutor",
    "IcebergExecutor",
    "SmartOrderRouter",
    "ExecutionResult",
    # Fill Simulation
    "SimpleFillModel",
    "QueueFillModel",
    "ImpactFillModel",
    "FillSimulationResult",
    # Hedge
    "HedgeSimulator",
    "HedgePosition",
    "HedgeTrade",
    "DeltaExposure",
    # Engine
    "ExecutionEngine",
    "ExecutionDecision",
    "EngineState",
]
