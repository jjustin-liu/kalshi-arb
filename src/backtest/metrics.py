"""
Performance Metrics Calculator.

Calculates key trading metrics for backtest evaluation:
- Risk-adjusted returns (Sharpe, Sortino)
- Drawdown analysis
- Trade statistics (win rate, profit factor)
- Execution quality (slippage, edge captured)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import math

import numpy as np
import pandas as pd

from src.data_feed.schemas import BacktestTrade, PerformanceMetrics, PnLAttribution


@dataclass
class EquityCurve:
    """Equity curve data."""
    timestamps: list[datetime] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    drawdown: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "equity": self.equity,
            "drawdown": self.drawdown,
        })


class MetricsCalculator:
    """
    Calculates comprehensive performance metrics.

    Key metrics:
    - Sharpe Ratio: Risk-adjusted return (excess return / volatility)
    - Sortino Ratio: Downside-risk adjusted return
    - Maximum Drawdown: Largest peak-to-trough decline
    - Win Rate: Percentage of profitable trades
    - Profit Factor: Gross profit / gross loss
    - Edge Captured: Realized edge / expected edge

    Usage:
        calculator = MetricsCalculator(initial_capital=10000)

        # Add trades
        for trade in trades:
            calculator.add_trade(trade)

        # Get metrics
        metrics = calculator.calculate()
    """

    # Annualization factor (trading days per year)
    TRADING_DAYS = 252
    RISK_FREE_RATE = 0.05  # 5% annual risk-free rate

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.05,
    ):
        """
        Initialize metrics calculator.

        Args:
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        self._trades: list[BacktestTrade] = []
        self._daily_pnl: dict[datetime, float] = {}
        self._equity_curve = EquityCurve()

        # Running state
        self._current_equity = initial_capital
        self._peak_equity = initial_capital
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

    def add_trade(self, trade: BacktestTrade):
        """
        Add a completed trade to metrics calculation.

        Args:
            trade: Completed backtest trade
        """
        self._trades.append(trade)

        # Update equity
        self._current_equity += trade.pnl

        # Track daily P&L
        day = trade.timestamp.date()
        day_dt = datetime(day.year, day.month, day.day)
        if day_dt not in self._daily_pnl:
            self._daily_pnl[day_dt] = 0.0
        self._daily_pnl[day_dt] += trade.pnl

        # Update peak and drawdown
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity

        drawdown = (self._peak_equity - self._current_equity) / self._peak_equity

        # Update equity curve
        self._equity_curve.timestamps.append(trade.timestamp)
        self._equity_curve.equity.append(self._current_equity)
        self._equity_curve.drawdown.append(drawdown)

        # Track time range
        if self._start_time is None:
            self._start_time = trade.timestamp
        self._end_time = trade.timestamp

    def calculate(self) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Returns:
            PerformanceMetrics dataclass with all metrics
        """
        if not self._trades:
            return self._empty_metrics()

        # Basic stats
        total_pnl = sum(t.pnl for t in self._trades)
        total_return = total_pnl / self.initial_capital

        # Duration
        duration_days = (self._end_time - self._start_time).days if self._start_time and self._end_time else 1
        duration_years = max(duration_days / 365.25, 1 / 365.25)

        # Annualized return
        annualized_return = (1 + total_return) ** (1 / duration_years) - 1

        # Daily returns for Sharpe/Sortino
        daily_returns = self._calculate_daily_returns()

        # Risk metrics
        sharpe = self._calculate_sharpe(daily_returns)
        sortino = self._calculate_sortino(daily_returns)
        max_dd, max_dd_duration = self._calculate_max_drawdown()

        # Trade statistics
        winners = [t for t in self._trades if t.pnl > 0]
        losers = [t for t in self._trades if t.pnl <= 0]

        win_rate = len(winners) / len(self._trades) if self._trades else 0
        avg_win = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl for t in losers]) if losers else 0

        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Execution quality
        fill_rate = self._calculate_fill_rate()
        avg_slippage = np.mean([t.slippage for t in self._trades])
        edge_captured = self._calculate_edge_captured()

        # Toxicity
        toxicity_saves = sum(
            1 for t in self._trades
            if hasattr(t, 'toxicity_filtered') and t.toxicity_filtered
        )

        return PerformanceMetrics(
            start_date=self._start_time or datetime.utcnow(),
            end_date=self._end_time or datetime.utcnow(),
            total_pnl=total_pnl,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            total_trades=len(self._trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            profit_factor=profit_factor,
            fill_rate=fill_rate,
            avg_slippage=float(avg_slippage),
            edge_captured=edge_captured,
            toxicity_saves=toxicity_saves,
            toxicity_precision=0.0,  # Would need counterfactual analysis
        )

    def _calculate_daily_returns(self) -> np.ndarray:
        """Calculate daily returns series."""
        if not self._daily_pnl:
            return np.array([0.0])

        # Sort by date
        sorted_days = sorted(self._daily_pnl.keys())

        # Calculate returns
        equity = self.initial_capital
        returns = []

        for day in sorted_days:
            daily_ret = self._daily_pnl[day] / equity
            returns.append(daily_ret)
            equity += self._daily_pnl[day]

        return np.array(returns)

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """
        Calculate annualized Sharpe ratio.

        Sharpe = (mean_return - risk_free) / std_dev * sqrt(252)
        """
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        daily_rf = self.risk_free_rate / self.TRADING_DAYS
        sharpe = (mean_return - daily_rf) / std_return * math.sqrt(self.TRADING_DAYS)

        return float(sharpe)

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """
        Calculate annualized Sortino ratio.

        Sortino = (mean_return - risk_free) / downside_std * sqrt(252)
        """
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        daily_rf = self.risk_free_rate / self.TRADING_DAYS

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < daily_rf]
        if len(downside_returns) < 2:
            return float('inf') if mean_return > daily_rf else 0.0

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return float('inf') if mean_return > daily_rf else 0.0

        sortino = (mean_return - daily_rf) / downside_std * math.sqrt(self.TRADING_DAYS)

        return float(sortino)

    def _calculate_max_drawdown(self) -> tuple[float, float]:
        """
        Calculate maximum drawdown and its duration.

        Returns:
            (max_drawdown_pct, duration_days) tuple
        """
        if not self._equity_curve.equity:
            return 0.0, 0.0

        equity = np.array(self._equity_curve.equity)
        timestamps = self._equity_curve.timestamps

        # Running maximum
        peak = np.maximum.accumulate(equity)

        # Drawdown series
        drawdown = (peak - equity) / peak

        # Maximum drawdown
        max_dd = float(np.max(drawdown))

        # Duration (time from peak to recovery or end)
        if max_dd > 0:
            max_dd_idx = np.argmax(drawdown)
            peak_idx = np.argmax(equity[:max_dd_idx + 1])

            # Find recovery (if any)
            recovery_idx = None
            for i in range(max_dd_idx, len(equity)):
                if equity[i] >= peak[max_dd_idx]:
                    recovery_idx = i
                    break

            if recovery_idx is not None:
                duration = (timestamps[recovery_idx] - timestamps[peak_idx]).days
            else:
                duration = (timestamps[-1] - timestamps[peak_idx]).days
        else:
            duration = 0.0

        return max_dd, float(duration)

    def _calculate_fill_rate(self) -> float:
        """Calculate fill rate (filled / attempted)."""
        # This would need signal data to calculate properly
        # For now, assume all trades represent filled orders
        return 1.0

    def _calculate_edge_captured(self) -> float:
        """
        Calculate edge captured (realized / expected).

        Edge captured = mean(realized_edge) / mean(signal_edge)
        """
        if not self._trades:
            return 0.0

        total_expected = sum(t.signal_edge for t in self._trades)
        total_realized = sum(t.realized_edge for t in self._trades)

        if total_expected == 0:
            return 0.0

        return total_realized / total_expected

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no trades."""
        now = datetime.utcnow()
        return PerformanceMetrics(
            start_date=now,
            end_date=now,
            total_pnl=0.0,
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            fill_rate=0.0,
            avg_slippage=0.0,
            edge_captured=0.0,
            toxicity_saves=0,
            toxicity_precision=0.0,
        )

    def get_equity_curve(self) -> EquityCurve:
        """Get equity curve data."""
        return self._equity_curve

    def get_attribution(self, period: str = "total") -> PnLAttribution:
        """
        Get P&L attribution breakdown.

        Args:
            period: "total", "daily", "weekly"

        Returns:
            PnLAttribution with component breakdown
        """
        if not self._trades:
            return PnLAttribution(
                timestamp=datetime.utcnow(),
                period=period,
                signal_pnl=0.0,
                execution_pnl=0.0,
                toxicity_saves=0.0,
                fees_paid=0.0,
                slippage_cost=0.0,
            )

        # Signal P&L: from correct directional calls
        signal_pnl = sum(
            t.signal_edge * t.quantity / 100
            for t in self._trades
        )

        # Execution P&L: difference from expected
        execution_pnl = sum(
            (t.realized_edge - t.signal_edge) * t.quantity / 100
            for t in self._trades
        )

        # Fees
        fees = sum(t.fees for t in self._trades)

        # Slippage
        slippage = sum(t.slippage * t.quantity / 100 for t in self._trades)

        return PnLAttribution(
            timestamp=datetime.utcnow(),
            period=period,
            signal_pnl=signal_pnl,
            execution_pnl=execution_pnl,
            toxicity_saves=0.0,  # Would need counterfactual
            fees_paid=fees,
            slippage_cost=slippage,
        )

    def reset(self):
        """Reset all state."""
        self._trades.clear()
        self._daily_pnl.clear()
        self._equity_curve = EquityCurve()
        self._current_equity = self.initial_capital
        self._peak_equity = self.initial_capital
        self._start_time = None
        self._end_time = None
