"""
Backtest Module.

Event-driven backtesting framework for strategy evaluation.

Components:
- EventEngine: Chronological event simulation
- FillModel: Realistic fill simulation (Simple, Queue, Impact)
- MetricsCalculator: Performance metrics (Sharpe, drawdown, etc.)
- BacktestEngine: Main orchestrator
- ReportGenerator: HTML/JSON report generation
- Optimizer: Grid search and walk-forward optimization

Usage:
    from src.backtest import BacktestEngine, BacktestConfig

    config = BacktestConfig(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        initial_capital=10000,
    )

    engine = BacktestEngine(config)
    result = engine.run()

    print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
    print(f"P&L: ${result.metrics.total_pnl:.2f}")
"""

from .config import BacktestConfig, OptimizationConfig
from .event_engine import EventEngine, Event, EventType
from .fill_model import (
    BaseFillModel,
    SimpleFillModel,
    QueueFillModel,
    ImpactFillModel,
    FillResult,
    create_fill_model,
)
from .metrics import MetricsCalculator, EquityCurve
from .backtest_engine import BacktestEngine, BacktestResult, MarketState
from .report import ReportGenerator
from .optimizer import (
    GridSearchOptimizer,
    WalkForwardOptimizer,
    ParameterSet,
    OptimizationResult,
    WalkForwardResult,
)

__all__ = [
    # Config
    "BacktestConfig",
    "OptimizationConfig",
    # Event Engine
    "EventEngine",
    "Event",
    "EventType",
    # Fill Models
    "BaseFillModel",
    "SimpleFillModel",
    "QueueFillModel",
    "ImpactFillModel",
    "FillResult",
    "create_fill_model",
    # Metrics
    "MetricsCalculator",
    "EquityCurve",
    # Backtest Engine
    "BacktestEngine",
    "BacktestResult",
    "MarketState",
    # Report
    "ReportGenerator",
    # Optimizer
    "GridSearchOptimizer",
    "WalkForwardOptimizer",
    "ParameterSet",
    "OptimizationResult",
    "WalkForwardResult",
]
