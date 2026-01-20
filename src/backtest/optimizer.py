"""
Parameter Optimizer for Backtest Strategies.

Provides:
- Grid search over parameter space
- Walk-forward optimization (train/test split)
- Result analysis and best parameter selection
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterator, Optional
import itertools

from .config import BacktestConfig, OptimizationConfig
from .backtest_engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class ParameterSet:
    """A set of parameters to test."""
    min_edge_threshold: float
    toxicity_threshold: float
    position_size: int


@dataclass
class OptimizationResult:
    """Result of a single optimization run."""
    parameters: ParameterSet
    train_result: BacktestResult
    test_result: Optional[BacktestResult]
    train_metric: float
    test_metric: Optional[float]


@dataclass
class WalkForwardResult:
    """Result of walk-forward optimization."""
    best_parameters: ParameterSet
    results: list[OptimizationResult]
    oos_performance: float  # Out-of-sample performance
    is_performance: float  # In-sample performance
    degradation: float  # IS to OOS degradation


class GridSearchOptimizer:
    """
    Grid search optimizer for backtest parameters.

    Tests all combinations of parameters and ranks by target metric.

    Usage:
        optimizer = GridSearchOptimizer(config)
        results = optimizer.run(base_config)
        best = optimizer.get_best_parameters()
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self._results: list[OptimizationResult] = []

    def generate_parameter_sets(self) -> Iterator[ParameterSet]:
        """
        Generate all parameter combinations to test.

        Yields:
            ParameterSet for each combination
        """
        cfg = self.config

        # Generate ranges
        min_edges = self._arange(*cfg.min_edge_range)
        toxicity_thresholds = self._arange(*cfg.toxicity_threshold_range)
        position_sizes = list(range(
            cfg.position_size_range[0],
            cfg.position_size_range[1] + 1,
            cfg.position_size_range[2],
        ))

        # All combinations
        for edge, tox, size in itertools.product(
            min_edges, toxicity_thresholds, position_sizes
        ):
            yield ParameterSet(
                min_edge_threshold=edge,
                toxicity_threshold=tox,
                position_size=size,
            )

    def _arange(self, start: float, stop: float, step: float) -> list[float]:
        """Generate range of floats."""
        result = []
        current = start
        while current <= stop:
            result.append(round(current, 6))
            current += step
        return result

    def run(self, base_config: BacktestConfig) -> list[OptimizationResult]:
        """
        Run grid search optimization.

        Args:
            base_config: Base backtest configuration

        Returns:
            List of results for each parameter set
        """
        self._results.clear()
        parameter_sets = list(self.generate_parameter_sets())
        total = len(parameter_sets)

        logger.info(f"Starting grid search with {total} parameter combinations")

        for i, params in enumerate(parameter_sets):
            logger.info(f"Testing {i+1}/{total}: edge={params.min_edge_threshold:.3f}, "
                       f"tox={params.toxicity_threshold:.2f}, size={params.position_size}")

            # Create config with these parameters
            config = BacktestConfig(
                start_date=base_config.start_date,
                end_date=base_config.end_date,
                initial_capital=base_config.initial_capital,
                data_dir=base_config.data_dir,
                fill_model=base_config.fill_model,
                min_edge_threshold=params.min_edge_threshold,
                max_position_per_market=params.position_size,
                toxicity_threshold=params.toxicity_threshold,
            )

            # Run backtest
            engine = BacktestEngine(config)
            result = engine.run()

            # Extract metric
            metric = self._extract_metric(result)

            # Check constraints
            if not self._check_constraints(result):
                logger.debug(f"  Constraints not met, skipping")
                continue

            opt_result = OptimizationResult(
                parameters=params,
                train_result=result,
                test_result=None,
                train_metric=metric,
                test_metric=None,
            )
            self._results.append(opt_result)

            logger.info(f"  {self.config.target_metric}={metric:.4f}, "
                       f"trades={result.metrics.total_trades}")

        # Sort by metric
        self._results.sort(key=lambda r: r.train_metric, reverse=True)

        return self._results

    def _extract_metric(self, result: BacktestResult) -> float:
        """Extract target metric from result."""
        m = result.metrics

        if self.config.target_metric == "sharpe":
            return m.sharpe_ratio
        elif self.config.target_metric == "sortino":
            return m.sortino_ratio
        elif self.config.target_metric == "total_pnl":
            return m.total_pnl
        elif self.config.target_metric == "profit_factor":
            return m.profit_factor
        else:
            return m.sharpe_ratio

    def _check_constraints(self, result: BacktestResult) -> bool:
        """Check if result meets constraints."""
        cfg = self.config

        if result.metrics.total_trades < cfg.min_trades:
            return False

        if result.metrics.max_drawdown > cfg.max_drawdown:
            return False

        return True

    def get_best_parameters(self) -> Optional[ParameterSet]:
        """Get best parameters from results."""
        if not self._results:
            return None
        return self._results[0].parameters

    def get_results(self) -> list[OptimizationResult]:
        """Get all optimization results."""
        return self._results


class WalkForwardOptimizer:
    """
    Walk-forward optimizer with rolling train/test windows.

    Tests strategy robustness by:
    1. Optimize on training window
    2. Test best params on out-of-sample window
    3. Roll forward and repeat

    This helps detect overfitting - if OOS performance is much worse
    than IS performance, the strategy may be overfit.

    Usage:
        optimizer = WalkForwardOptimizer(config)
        result = optimizer.run(base_config)
        print(f"OOS Sharpe: {result.oos_performance:.2f}")
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize walk-forward optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()

    def run(self, base_config: BacktestConfig) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            base_config: Base configuration (dates will be overridden)

        Returns:
            WalkForwardResult with performance analysis
        """
        cfg = self.config
        results: list[OptimizationResult] = []

        # Calculate windows
        total_days = (base_config.end_date - base_config.start_date).days
        window_size = cfg.train_days + cfg.test_days
        num_windows = (total_days - window_size) // cfg.step_days + 1

        logger.info(f"Walk-forward: {num_windows} windows, "
                   f"train={cfg.train_days}d, test={cfg.test_days}d")

        current_start = base_config.start_date

        for window in range(num_windows):
            train_start = current_start
            train_end = train_start + timedelta(days=cfg.train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=cfg.test_days)

            if test_end > base_config.end_date:
                break

            logger.info(f"Window {window + 1}: train {train_start} to {train_end}, "
                       f"test {test_start} to {test_end}")

            # Optimize on training period
            train_config = BacktestConfig(
                start_date=train_start,
                end_date=train_end,
                initial_capital=base_config.initial_capital,
                data_dir=base_config.data_dir,
                fill_model=base_config.fill_model,
            )

            grid_opt = GridSearchOptimizer(self.config)
            grid_opt.run(train_config)

            best_params = grid_opt.get_best_parameters()
            if best_params is None:
                logger.warning(f"  No valid parameters found for window {window + 1}")
                current_start += timedelta(days=cfg.step_days)
                continue

            # Test on out-of-sample period
            test_config = BacktestConfig(
                start_date=test_start,
                end_date=test_end,
                initial_capital=base_config.initial_capital,
                data_dir=base_config.data_dir,
                fill_model=base_config.fill_model,
                min_edge_threshold=best_params.min_edge_threshold,
                max_position_per_market=best_params.position_size,
                toxicity_threshold=best_params.toxicity_threshold,
            )

            test_engine = BacktestEngine(test_config)
            test_result = test_engine.run()
            test_metric = self._extract_metric(test_result)

            # Get training result for comparison
            train_results = grid_opt.get_results()
            train_metric = train_results[0].train_metric if train_results else 0.0

            result = OptimizationResult(
                parameters=best_params,
                train_result=train_results[0].train_result if train_results else None,
                test_result=test_result,
                train_metric=train_metric,
                test_metric=test_metric,
            )
            results.append(result)

            logger.info(f"  Best params: edge={best_params.min_edge_threshold:.3f}, "
                       f"IS {self.config.target_metric}={train_metric:.4f}, "
                       f"OOS {self.config.target_metric}={test_metric:.4f}")

            # Roll forward
            current_start += timedelta(days=cfg.step_days)

        # Aggregate results
        if not results:
            return WalkForwardResult(
                best_parameters=ParameterSet(0.01, 0.6, 50),
                results=[],
                oos_performance=0.0,
                is_performance=0.0,
                degradation=0.0,
            )

        # Calculate aggregate performance
        is_metrics = [r.train_metric for r in results if r.train_metric]
        oos_metrics = [r.test_metric for r in results if r.test_metric]

        avg_is = sum(is_metrics) / len(is_metrics) if is_metrics else 0
        avg_oos = sum(oos_metrics) / len(oos_metrics) if oos_metrics else 0
        degradation = (avg_is - avg_oos) / avg_is if avg_is > 0 else 0

        # Best parameters = most frequently selected
        param_counts: dict[tuple, int] = {}
        for r in results:
            key = (
                r.parameters.min_edge_threshold,
                r.parameters.toxicity_threshold,
                r.parameters.position_size,
            )
            param_counts[key] = param_counts.get(key, 0) + 1

        best_key = max(param_counts, key=param_counts.get)
        best_params = ParameterSet(*best_key)

        return WalkForwardResult(
            best_parameters=best_params,
            results=results,
            oos_performance=avg_oos,
            is_performance=avg_is,
            degradation=degradation,
        )

    def _extract_metric(self, result: BacktestResult) -> float:
        """Extract target metric from result."""
        m = result.metrics

        if self.config.target_metric == "sharpe":
            return m.sharpe_ratio
        elif self.config.target_metric == "sortino":
            return m.sortino_ratio
        elif self.config.target_metric == "total_pnl":
            return m.total_pnl
        elif self.config.target_metric == "profit_factor":
            return m.profit_factor
        else:
            return m.sharpe_ratio
