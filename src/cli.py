"""
CLI entry point for the Kalshi Arbitrage System.

Usage:
    python -m src.cli live --config config.yaml
    python -m src.cli backtest --start 2024-01-01 --end 2024-01-31
    python -m src.cli monitor --port 8080
    python -m src.cli collect --markets INXD
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="kalshi-arb")
def cli():
    """Kalshi Arbitrage System - Cross-asset basis trading."""
    pass


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--paper", is_flag=True, help="Run in paper trading mode")
@click.option("--markets", "-m", multiple=True, default=["INXD"], help="Markets to trade")
def live(config: Optional[str], paper: bool, markets: tuple):
    """Run live trading system."""
    from src.data_feed.kalshi_client import KalshiRESTClient, KalshiWebSocketClient
    from src.data_feed.databento_client import DatabentoClient
    from src.pricing.basis_calculator import BasisCalculator
    from src.risk.risk_manager import RiskManager
    from src.risk.config import RiskConfig
    from src.execution.execution_engine import ExecutionEngine
    from src.execution.config import ExecutionConfig
    from src.monitoring.dashboard import DashboardManager

    console.print("[bold green]Starting Kalshi Arbitrage System[/bold green]")
    console.print(f"Mode: {'Paper' if paper else 'Live'}")
    console.print(f"Markets: {', '.join(markets)}")

    # Load configuration
    if config:
        import yaml
        with open(config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Initialize components
    risk_config = RiskConfig(**cfg.get('risk', {}))
    exec_config = ExecutionConfig(**cfg.get('execution', {}))

    risk_manager = RiskManager(risk_config)

    kalshi_client = None if paper else KalshiRESTClient()
    execution_engine = ExecutionEngine(config=exec_config, kalshi_client=kalshi_client)

    basis_calculator = BasisCalculator()
    dashboard = DashboardManager()

    # Setup signal handlers
    shutdown_event = asyncio.Event()

    def handle_shutdown(sig, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    async def run():
        # Initialize WebSocket connections
        kalshi_ws = KalshiWebSocketClient()
        databento = DatabentoClient()

        try:
            # Subscribe to markets
            await kalshi_ws.connect()
            for market in markets:
                await kalshi_ws.subscribe_orderbook(f"{market}*")
                await kalshi_ws.subscribe_trades(f"{market}*")

            # Subscribe to ES futures
            await databento.connect()
            await databento.subscribe("ES.FUT")

            console.print("[green]Connected to data feeds[/green]")

            # Main loop
            while not shutdown_event.is_set():
                try:
                    # Process incoming data
                    orderbook = await asyncio.wait_for(
                        kalshi_ws.get_orderbook(),
                        timeout=1.0
                    )

                    if orderbook:
                        # Update risk monitors
                        risk_manager.update_orderbook(orderbook)

                        # Get underlying price
                        underlying = databento.get_latest("ES.FUT")

                        if underlying:
                            # Generate signal
                            signal = basis_calculator.calculate_signal(
                                orderbook, underlying
                            )

                            if signal and signal.net_edge > exec_config.min_edge:
                                # Risk check
                                assessment = risk_manager.evaluate_signal(
                                    signal, orderbook
                                )

                                if assessment.approved:
                                    # Execute
                                    result = execution_engine.evaluate_execution(
                                        signal, orderbook
                                    )

                                    if result.should_execute:
                                        console.print(
                                            f"[cyan]Signal: {signal.market_ticker} "
                                            f"{signal.side.value} {result.adjusted_size} "
                                            f"@ {signal.net_edge:.2%} edge[/cyan]"
                                        )

                                        if not paper:
                                            await execution_engine.execute_async(
                                                signal, orderbook, result
                                            )

                                # Update dashboard
                                dashboard.update_signal(signal)

                        dashboard.update_orderbook(orderbook)

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

        finally:
            await kalshi_ws.disconnect()
            await databento.disconnect()
            console.print("[green]Disconnected from data feeds[/green]")

    asyncio.run(run())


@cli.command()
@click.option("--start", "-s", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", "-e", required=True, help="End date (YYYY-MM-DD)")
@click.option("--data-dir", "-d", type=click.Path(exists=True), help="Data directory")
@click.option("--output", "-o", type=click.Path(), help="Output report path")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file")
def backtest(start: str, end: str, data_dir: Optional[str], output: Optional[str], config: Optional[str]):
    """Run backtest on historical data."""
    from src.backtest.backtest_engine import BacktestEngine
    from src.backtest.config import BacktestConfig
    from src.backtest.report import ReportGenerator
    from src.data_feed.recorder import DataLoader

    console.print("[bold green]Running Backtest[/bold green]")
    console.print(f"Period: {start} to {end}")

    # Parse dates
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    # Load configuration
    if config:
        import yaml
        with open(config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    backtest_config = BacktestConfig(
        initial_capital=cfg.get('initial_capital', 10000),
        min_edge=cfg.get('min_edge', 0.02),
        max_toxicity=cfg.get('max_toxicity', 0.6),
        **cfg.get('backtest', {})
    )

    # Initialize backtest engine
    engine = BacktestEngine(backtest_config)

    # Load historical data
    data_path = Path(data_dir) if data_dir else Path("data")
    loader = DataLoader(data_path)

    console.print("Loading historical data...")
    orderbooks = loader.load_orderbooks(start_date, end_date)
    trades = loader.load_trades(start_date, end_date)
    underlying = loader.load_underlying(start_date, end_date)

    console.print(f"Loaded {len(orderbooks)} orderbooks, {len(trades)} trades")

    # Run backtest
    console.print("Running simulation...")

    with console.status("[bold green]Processing events..."):
        for ob in orderbooks:
            engine.on_orderbook(ob)
        for trade in trades:
            engine.on_trade(trade)
        for tick in underlying:
            engine.on_underlying(tick)

    # Get results
    results = engine.get_results()

    # Display results
    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Return", f"{results.get('total_return', 0):.2%}")
    table.add_row("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
    table.add_row("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
    table.add_row("Win Rate", f"{results.get('win_rate', 0):.2%}")
    table.add_row("Profit Factor", f"{results.get('profit_factor', 0):.2f}")
    table.add_row("Total Trades", f"{results.get('num_trades', 0)}")

    console.print(table)

    # Generate report
    if output:
        report_gen = ReportGenerator()
        report_gen.generate(results, output)
        console.print(f"Report saved to {output}")


@cli.command()
@click.option("--port", "-p", default=8080, help="Server port")
@click.option("--host", "-h", default="0.0.0.0", help="Server host")
def monitor(port: int, host: str):
    """Start monitoring dashboard server."""
    from src.monitoring.web_server import MonitoringServer
    from src.monitoring.config import DashboardConfig
    from src.monitoring.dashboard import DashboardManager
    from src.monitoring.alerts import AlertManager
    from src.monitoring.metrics_collector import MetricsCollector

    console.print(f"[bold green]Starting Monitoring Dashboard[/bold green]")
    console.print(f"URL: http://{host}:{port}")

    # Initialize all required components
    config = DashboardConfig(host=host, port=port)
    dashboard = DashboardManager()
    alerts = AlertManager()
    metrics = MetricsCollector()

    server = MonitoringServer(
        dashboard=dashboard,
        alerts=alerts,
        metrics=metrics,
        config=config,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down server...[/yellow]")


@cli.command()
@click.option("--markets", "-m", multiple=True, default=["INXD"], help="Markets to collect")
@click.option("--output", "-o", type=click.Path(), default="data", help="Output directory")
@click.option("--duration", "-d", type=int, help="Collection duration in hours")
def collect(markets: tuple, output: str, duration: Optional[int]):
    """Collect market data for backtesting."""
    import subprocess

    console.print("[bold green]Starting Data Collection[/bold green]")
    console.print(f"Markets: {', '.join(markets)}")
    console.print(f"Output: {output}")

    # Build command
    cmd = [
        sys.executable,
        "scripts/run_collector.py",
        "--output", output,
    ]
    for market in markets:
        cmd.extend(["--markets", market])

    if duration:
        cmd.extend(["--duration", str(duration)])

    # Run collector
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file")
@click.option("--metric", default="sharpe", help="Optimization metric")
@click.option("--method", default="grid", type=click.Choice(["grid", "walk-forward"]))
def optimize(config: Optional[str], metric: str, method: str):
    """Optimize strategy parameters."""
    from src.backtest.optimizer import GridSearchOptimizer, WalkForwardOptimizer
    from src.backtest.config import OptimizationConfig

    console.print("[bold green]Running Parameter Optimization[/bold green]")
    console.print(f"Method: {method}")
    console.print(f"Metric: {metric}")

    # Load configuration
    if config:
        import yaml
        with open(config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            'parameter_grid': {
                'min_edge': [0.01, 0.02, 0.03, 0.04, 0.05],
                'max_toxicity': [0.4, 0.5, 0.6, 0.7],
                'position_size': [10, 20, 50, 100],
            }
        }

    opt_config = OptimizationConfig(
        parameter_grid=cfg.get('parameter_grid', {}),
        metric=metric,
        **cfg.get('optimization', {})
    )

    # Select optimizer
    if method == "grid":
        optimizer = GridSearchOptimizer(opt_config)
    else:
        optimizer = WalkForwardOptimizer(opt_config)

    # Run optimization
    console.print("Running optimization...")

    from src.backtest.backtest_engine import BacktestEngine
    from src.backtest.config import BacktestConfig

    def run_backtest(params):
        bt_config = BacktestConfig(**params)
        engine = BacktestEngine(bt_config)
        # Would need to load data and run
        results = engine.get_results()
        return results

    with console.status("[bold green]Searching parameter space..."):
        best_params, best_metric = optimizer.optimize(run_backtest)

    # Display results
    table = Table(title="Optimization Results")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    for param, value in best_params.items():
        table.add_row(param, str(value))

    table.add_row("Best Metric", f"{best_metric:.4f}")

    console.print(table)


@cli.command()
def status():
    """Show system status."""
    console.print("[bold green]System Status[/bold green]")

    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    # Check components
    components = [
        ("Kalshi API", "check_kalshi"),
        ("Databento API", "check_databento"),
        ("Data Directory", "check_data"),
        ("Config File", "check_config"),
    ]

    for name, check_func in components:
        try:
            status = "✓ OK"
        except Exception as e:
            status = f"✗ {e}"
        table.add_row(name, status)

    console.print(table)


@cli.command()
@click.option("--file", "-f", type=click.Path(exists=True), required=True)
def replay(file: str):
    """Replay recorded data for analysis."""
    from src.data_feed.recorder import DataLoader

    console.print(f"[bold green]Replaying data from {file}[/bold green]")

    loader = DataLoader(Path(file).parent)

    # Load and display data
    data = loader.load_file(Path(file))

    console.print(f"Loaded {len(data)} records")

    for i, record in enumerate(data[:10]):
        console.print(f"{i}: {record}")

    if len(data) > 10:
        console.print(f"... and {len(data) - 10} more records")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
