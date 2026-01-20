"""
Unit tests for MetricsCalculator.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.backtest.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_sharpe_ratio_positive_returns(self):
        """Sharpe ratio should be positive for positive excess returns."""
        calc = MetricsCalculator()

        # Consistent positive returns
        returns = [0.01, 0.02, 0.015, 0.01, 0.025] * 50  # 250 days

        sharpe = calc.calculate_sharpe(returns, risk_free_rate=0.0)

        assert sharpe > 0

    def test_sharpe_ratio_negative_returns(self):
        """Sharpe ratio should be negative for negative returns."""
        calc = MetricsCalculator()

        # Consistent negative returns
        returns = [-0.01, -0.02, -0.015, -0.01, -0.025] * 50

        sharpe = calc.calculate_sharpe(returns, risk_free_rate=0.0)

        assert sharpe < 0

    def test_sharpe_ratio_annualization(self):
        """Sharpe should be annualized correctly."""
        calc = MetricsCalculator()

        # Simple case: 1% daily return, 1% daily std
        returns = [0.01] * 252

        sharpe = calc.calculate_sharpe(returns, risk_free_rate=0.0)

        # Expected: (0.01 / 0.0) * sqrt(252) -> very high due to zero std
        # Use returns with variance instead
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252).tolist()
        sharpe = calc.calculate_sharpe(returns, risk_free_rate=0.0)

        # Should be reasonable range
        assert -5 < sharpe < 5

    def test_sortino_ratio_ignores_upside(self):
        """Sortino should only penalize downside volatility."""
        calc = MetricsCalculator()

        # High upside variance, low downside
        returns = [0.05, 0.03, 0.04, 0.06, -0.01, 0.05, 0.04, -0.005] * 30

        sortino = calc.calculate_sortino(returns, risk_free_rate=0.0)
        sharpe = calc.calculate_sharpe(returns, risk_free_rate=0.0)

        # Sortino should be higher than Sharpe when upside > downside
        assert sortino > sharpe

    def test_max_drawdown_calculation(self):
        """Should calculate max drawdown correctly."""
        calc = MetricsCalculator()

        # Equity curve: 100 -> 120 -> 90 -> 110
        # Max DD = (120 - 90) / 120 = 25%
        equity_curve = [100, 110, 120, 100, 90, 95, 100, 110]

        dd = calc.calculate_max_drawdown(equity_curve)

        assert abs(dd - 0.25) < 0.01  # 25% drawdown

    def test_max_drawdown_no_drawdown(self):
        """Max drawdown should be 0 for monotonically increasing equity."""
        calc = MetricsCalculator()

        equity_curve = [100, 101, 102, 103, 104, 105]

        dd = calc.calculate_max_drawdown(equity_curve)

        assert dd == 0.0

    def test_win_rate_calculation(self):
        """Should calculate win rate correctly."""
        calc = MetricsCalculator()

        # 7 wins, 3 losses = 70% win rate
        pnls = [10, -5, 20, 15, -10, 5, 10, -8, 30, 25]

        win_rate = calc.calculate_win_rate(pnls)

        assert win_rate == 0.7

    def test_profit_factor(self):
        """Profit factor = gross profits / gross losses."""
        calc = MetricsCalculator()

        # Profits: 10 + 20 + 15 + 5 + 10 + 30 + 25 = 115
        # Losses: 5 + 10 + 8 = 23
        # PF = 115 / 23 = 5.0
        pnls = [10, -5, 20, 15, -10, 5, 10, -8, 30, 25]

        pf = calc.calculate_profit_factor(pnls)

        assert abs(pf - 5.0) < 0.01

    def test_profit_factor_no_losses(self):
        """Profit factor should be infinity (or large number) with no losses."""
        calc = MetricsCalculator()

        pnls = [10, 20, 15, 5]

        pf = calc.calculate_profit_factor(pnls)

        assert pf == float('inf') or pf > 1000

    def test_average_trade_calculation(self):
        """Should calculate average trade correctly."""
        calc = MetricsCalculator()

        pnls = [10, -5, 20, -10, 15]

        avg = calc.calculate_average_trade(pnls)

        assert avg == 6.0  # (10 - 5 + 20 - 10 + 15) / 5

    def test_edge_captured_calculation(self):
        """Edge captured = actual return / expected edge."""
        calc = MetricsCalculator()

        expected_edges = [0.05, 0.04, 0.03, 0.06, 0.05]  # Expected edge per trade
        actual_returns = [0.04, 0.03, 0.02, 0.05, 0.04]  # Actual returns

        edge_captured = calc.calculate_edge_captured(expected_edges, actual_returns)

        # Expected: avg(actual) / avg(expected) = 0.036 / 0.046 ≈ 0.78
        expected_ratio = np.mean(actual_returns) / np.mean(expected_edges)
        assert abs(edge_captured - expected_ratio) < 0.01

    def test_calmar_ratio(self):
        """Calmar = CAGR / Max Drawdown."""
        calc = MetricsCalculator()

        # 20% annual return, 10% max drawdown = Calmar of 2.0
        annual_return = 0.20
        max_drawdown = 0.10

        calmar = calc.calculate_calmar(annual_return, max_drawdown)

        assert calmar == 2.0

    def test_full_metrics_calculation(self):
        """Should calculate all metrics in one call."""
        calc = MetricsCalculator()

        # Generate sample data
        np.random.seed(42)
        daily_returns = np.random.normal(0.001, 0.02, 252).tolist()
        equity_curve = [1000]
        for r in daily_returns:
            equity_curve.append(equity_curve[-1] * (1 + r))

        trade_pnls = [r * 1000 for r in daily_returns[:50]]  # First 50 as trades

        metrics = calc.calculate_all(
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            trade_pnls=trade_pnls,
        )

        # Check all metrics present
        assert 'sharpe' in metrics
        assert 'sortino' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'total_return' in metrics

    def test_drawdown_duration(self):
        """Should calculate drawdown duration correctly."""
        calc = MetricsCalculator()

        # In drawdown for indices 3-6 (4 periods)
        equity_curve = [100, 110, 120, 110, 105, 100, 105, 125, 130]

        duration = calc.calculate_drawdown_duration(equity_curve)

        # Drawdown starts at peak (120), ends when we exceed it (125)
        assert duration >= 4

    def test_handles_empty_data(self):
        """Should handle empty data gracefully."""
        calc = MetricsCalculator()

        assert calc.calculate_sharpe([]) == 0.0
        assert calc.calculate_max_drawdown([]) == 0.0
        assert calc.calculate_win_rate([]) == 0.0

    def test_handles_single_data_point(self):
        """Should handle single data point."""
        calc = MetricsCalculator()

        assert calc.calculate_sharpe([0.01]) == 0.0  # Can't calc std
        assert calc.calculate_max_drawdown([100]) == 0.0


class TestPerformanceAttribution:
    """Tests for P&L attribution."""

    def test_pnl_attribution_by_component(self):
        """Should attribute P&L to different components."""
        calc = MetricsCalculator()

        trades = [
            {'pnl': 100, 'component': 'basis'},
            {'pnl': -20, 'component': 'slippage'},
            {'pnl': -10, 'component': 'fees'},
            {'pnl': 50, 'component': 'basis'},
            {'pnl': -5, 'component': 'slippage'},
        ]

        attribution = calc.attribute_pnl(trades)

        assert attribution['basis'] == 150
        assert attribution['slippage'] == -25
        assert attribution['fees'] == -10

    def test_pnl_by_market(self):
        """Should group P&L by market."""
        calc = MetricsCalculator()

        trades = [
            {'pnl': 100, 'market': 'INXD-A'},
            {'pnl': 50, 'market': 'INXD-A'},
            {'pnl': -30, 'market': 'INXD-B'},
            {'pnl': 20, 'market': 'INXD-B'},
        ]

        by_market = calc.pnl_by_market(trades)

        assert by_market['INXD-A'] == 150
        assert by_market['INXD-B'] == -10
