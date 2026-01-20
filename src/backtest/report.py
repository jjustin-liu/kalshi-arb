"""
Backtest Report Generation.

Creates HTML and JSON reports from backtest results for analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .backtest_engine import BacktestResult
from .metrics import EquityCurve


class ReportGenerator:
    """
    Generates backtest reports in various formats.

    Formats:
    - JSON: Machine-readable, for further analysis
    - HTML: Human-readable with charts

    Usage:
        generator = ReportGenerator(result)
        generator.save_json("report.json")
        generator.save_html("report.html")
    """

    def __init__(self, result: BacktestResult):
        """
        Initialize report generator.

        Args:
            result: Backtest result to report on
        """
        self.result = result

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "config": {
                "start_date": str(self.result.config.start_date),
                "end_date": str(self.result.config.end_date),
                "initial_capital": self.result.config.initial_capital,
                "fill_model": self.result.config.fill_model,
                "min_edge_threshold": self.result.config.min_edge_threshold,
            },
            "metrics": {
                "total_pnl": self.result.metrics.total_pnl,
                "total_return": self.result.metrics.total_return,
                "annualized_return": self.result.metrics.annualized_return,
                "sharpe_ratio": self.result.metrics.sharpe_ratio,
                "sortino_ratio": self.result.metrics.sortino_ratio,
                "max_drawdown": self.result.metrics.max_drawdown,
                "max_drawdown_duration": self.result.metrics.max_drawdown_duration,
                "total_trades": self.result.metrics.total_trades,
                "winning_trades": self.result.metrics.winning_trades,
                "losing_trades": self.result.metrics.losing_trades,
                "win_rate": self.result.metrics.win_rate,
                "avg_win": self.result.metrics.avg_win,
                "avg_loss": self.result.metrics.avg_loss,
                "profit_factor": self.result.metrics.profit_factor,
                "fill_rate": self.result.metrics.fill_rate,
                "avg_slippage": self.result.metrics.avg_slippage,
                "edge_captured": self.result.metrics.edge_captured,
            },
            "attribution": {
                "signal_pnl": self.result.attribution.signal_pnl,
                "execution_pnl": self.result.attribution.execution_pnl,
                "toxicity_saves": self.result.attribution.toxicity_saves,
                "fees_paid": self.result.attribution.fees_paid,
                "slippage_cost": self.result.attribution.slippage_cost,
                "total_pnl": self.result.attribution.total_pnl,
            },
            "summary": {
                "signals_generated": self.result.signals_generated,
                "signals_traded": self.result.signals_traded,
                "signals_filtered": self.result.signals_filtered,
            },
            "equity_curve": {
                "timestamps": [t.isoformat() for t in self.result.equity_curve.timestamps],
                "equity": self.result.equity_curve.equity,
                "drawdown": self.result.equity_curve.drawdown,
            },
        }

    def save_json(self, path: str):
        """Save report as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_html(self, path: str):
        """Save report as HTML with embedded charts."""
        html = self._generate_html()
        with open(path, "w") as f:
            f.write(html)

    def _generate_html(self) -> str:
        """Generate HTML report."""
        m = self.result.metrics
        a = self.result.attribution

        # Prepare chart data
        equity_data = json.dumps(self.result.equity_curve.equity)
        drawdown_data = json.dumps(self.result.equity_curve.drawdown)
        labels = json.dumps([
            t.strftime("%Y-%m-%d %H:%M")
            for t in self.result.equity_curve.timestamps
        ])

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 0; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; }}
    </style>
</head>
<body>
    <h1>Backtest Report</h1>
    <p>Period: {self.result.config.start_date} to {self.result.config.end_date}</p>

    <div class="card">
        <h2>Key Metrics</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value {'positive' if m.total_pnl >= 0 else 'negative'}">${m.total_pnl:,.2f}</div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if m.total_return >= 0 else 'negative'}">{m.total_return:.1%}</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{m.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value">{m.sortino_ratio:.2f}</div>
                <div class="metric-label">Sortino Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">{m.max_drawdown:.1%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric">
                <div class="metric-value">{m.win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{m.profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric">
                <div class="metric-value">{m.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Equity Curve</h2>
        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>
    </div>

    <div class="card">
        <h2>Drawdown</h2>
        <div class="chart-container">
            <canvas id="drawdownChart"></canvas>
        </div>
    </div>

    <div class="card">
        <h2>P&L Attribution</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Amount</th>
            </tr>
            <tr>
                <td>Signal P&L</td>
                <td class="{'positive' if a.signal_pnl >= 0 else 'negative'}">${a.signal_pnl:,.2f}</td>
            </tr>
            <tr>
                <td>Execution P&L</td>
                <td class="{'positive' if a.execution_pnl >= 0 else 'negative'}">${a.execution_pnl:,.2f}</td>
            </tr>
            <tr>
                <td>Fees Paid</td>
                <td class="negative">-${a.fees_paid:,.2f}</td>
            </tr>
            <tr>
                <td>Slippage Cost</td>
                <td class="negative">-${a.slippage_cost:,.2f}</td>
            </tr>
            <tr style="font-weight: bold;">
                <td>Total</td>
                <td class="{'positive' if a.total_pnl >= 0 else 'negative'}">${a.total_pnl:,.2f}</td>
            </tr>
        </table>
    </div>

    <div class="card">
        <h2>Trade Statistics</h2>
        <table>
            <tr>
                <td>Winning Trades</td>
                <td>{m.winning_trades}</td>
            </tr>
            <tr>
                <td>Losing Trades</td>
                <td>{m.losing_trades}</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td class="positive">${m.avg_win:,.2f}</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td class="negative">${m.avg_loss:,.2f}</td>
            </tr>
            <tr>
                <td>Average Slippage</td>
                <td>{m.avg_slippage:.4f}</td>
            </tr>
            <tr>
                <td>Edge Captured</td>
                <td>{m.edge_captured:.1%}</td>
            </tr>
        </table>
    </div>

    <div class="card">
        <h2>Signal Analysis</h2>
        <table>
            <tr>
                <td>Signals Generated</td>
                <td>{self.result.signals_generated}</td>
            </tr>
            <tr>
                <td>Signals Traded</td>
                <td>{self.result.signals_traded}</td>
            </tr>
            <tr>
                <td>Signals Filtered (Toxicity/Risk)</td>
                <td>{self.result.signals_filtered}</td>
            </tr>
            <tr>
                <td>Trade Rate</td>
                <td>{self.result.signals_traded / max(1, self.result.signals_generated):.1%}</td>
            </tr>
        </table>
    </div>

    <script>
        // Equity Chart
        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                labels: {labels},
                datasets: [{{
                    label: 'Equity',
                    data: {equity_data},
                    borderColor: '#007bff',
                    fill: false,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: false
                    }}
                }}
            }}
        }});

        // Drawdown Chart
        new Chart(document.getElementById('drawdownChart'), {{
            type: 'line',
            data: {{
                labels: {labels},
                datasets: [{{
                    label: 'Drawdown',
                    data: {drawdown_data},
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    fill: true,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        reverse: true,
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        return html
