"""Backtest module configuration."""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.

    Controls data sources, simulation parameters, and trading rules.
    """

    # Time range
    start_date: date = field(default_factory=lambda: date(2024, 1, 1))
    end_date: date = field(default_factory=lambda: date(2024, 1, 31))

    # Capital
    initial_capital: float = 10000.0

    # Data settings
    data_dir: str = "data/raw"

    # Fill model
    fill_model: str = "simple"  # "simple", "queue", "impact"

    # Trading parameters
    min_edge_threshold: float = 0.01  # 100 bps minimum edge
    max_position_per_market: int = 100
    max_total_position: int = 500

    # Toxicity settings
    toxicity_enabled: bool = True
    toxicity_threshold: float = 0.6

    # Fees
    taker_fee_per_contract: float = 0.07  # 7 cents

    # Output
    output_dir: str = "data/backtest"
    save_trades: bool = True
    save_signals: bool = True


@dataclass
class OptimizationConfig:
    """
    Configuration for parameter optimization.

    Defines parameter ranges and optimization method.
    """

    # Parameters to optimize
    min_edge_range: tuple[float, float, float] = (0.005, 0.03, 0.005)  # (min, max, step)
    toxicity_threshold_range: tuple[float, float, float] = (0.4, 0.8, 0.1)
    position_size_range: tuple[int, int, int] = (10, 100, 10)

    # Walk-forward settings
    train_days: int = 60
    test_days: int = 20
    step_days: int = 20  # How much to advance between windows

    # Optimization target
    target_metric: str = "sharpe"  # "sharpe", "sortino", "total_pnl", "profit_factor"

    # Constraints
    min_trades: int = 30  # Minimum trades for valid result
    max_drawdown: float = 0.10  # Maximum allowable drawdown
