"""
Combined data collector - runs Kalshi and Databento collectors together.

This is the main entry point for data collection. It runs both collectors
concurrently and provides unified status reporting and graceful shutdown.

Run this script for 1-2 weeks before backtesting to build sufficient history.

Usage:
    python scripts/run_collector.py --markets INXD --env demo
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.data_feed.kalshi_client import KalshiRESTClient, KalshiWebSocketClient
from src.data_feed.databento_client import DatabentoClient
from src.data_feed.recorder import DataRecorder
from src.data_feed.schemas import KalshiOrderbook, KalshiTrade, UnderlyingTick

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("collector.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
console = Console()


class CombinedCollector:
    """
    Runs Kalshi and Databento collectors concurrently.

    Provides:
    - Unified data recording to shared Parquet files
    - Combined status display showing both feeds
    - Graceful shutdown handling
    - Disk usage monitoring
    """

    def __init__(
        self,
        kalshi_api_key: str,
        kalshi_api_secret: str,
        databento_api_key: str,
        kalshi_env: str = "demo",
        data_dir: str = "data/raw",
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Shared recorder for all data
        self.recorder = DataRecorder(data_dir)

        # Kalshi clients
        self.kalshi_rest = KalshiRESTClient(
            kalshi_api_key, kalshi_api_secret, kalshi_env
        )
        self.kalshi_ws = KalshiWebSocketClient(
            kalshi_api_key, kalshi_api_secret, kalshi_env
        )

        # Databento client
        self.databento = DatabentoClient(databento_api_key)

        self._running = False
        self._stats = {
            "kalshi_orderbooks": 0,
            "kalshi_trades": 0,
            "databento_ticks": 0,
            "start_time": None,
            "last_es_price": None,
            "last_kalshi_update": None,
            "last_databento_update": None,
        }

    def _on_kalshi_orderbook(self, orderbook: KalshiOrderbook):
        """Handle Kalshi orderbook update."""
        self.recorder.record_kalshi_orderbook(orderbook)
        self._stats["kalshi_orderbooks"] += 1
        self._stats["last_kalshi_update"] = datetime.utcnow()

    def _on_kalshi_trade(self, trade: KalshiTrade):
        """Handle Kalshi trade."""
        self.recorder.record_kalshi_trade(trade)
        self._stats["kalshi_trades"] += 1
        self._stats["last_kalshi_update"] = datetime.utcnow()

    def _on_databento_tick(self, tick: UnderlyingTick):
        """Handle Databento tick."""
        self.recorder.record_underlying_tick(tick)
        self._stats["databento_ticks"] += 1
        self._stats["last_es_price"] = tick.price
        self._stats["last_databento_update"] = datetime.utcnow()

    async def discover_kalshi_markets(self, series_ticker: str) -> list[str]:
        """Discover active Kalshi markets."""
        markets = []
        cursor = None

        while True:
            batch, cursor = await self.kalshi_rest.get_markets(
                series_ticker=series_ticker,
                status="open",
                cursor=cursor,
            )
            markets.extend(batch)
            if not cursor:
                break

        tickers = [m.ticker for m in markets]
        logger.info(f"Discovered {len(tickers)} active Kalshi markets")
        return tickers

    async def _run_kalshi_collector(self, market_tickers: list[str]):
        """Run Kalshi WebSocket collector."""
        try:
            # Register callbacks
            self.kalshi_ws.on_orderbook(self._on_kalshi_orderbook)
            self.kalshi_ws.on_trade(self._on_kalshi_trade)

            # Connect and subscribe
            await self.kalshi_ws.connect()
            for ticker in market_tickers:
                await self.kalshi_ws.subscribe_orderbook(ticker)
                await self.kalshi_ws.subscribe_trades(ticker)
                await asyncio.sleep(0.1)

            logger.info(f"Kalshi: Subscribed to {len(market_tickers)} markets")

            # Listen for messages
            async for _ in self.kalshi_ws.listen():
                if not self._running:
                    break

        except Exception as e:
            logger.error(f"Kalshi collector error: {e}")
        finally:
            await self.kalshi_ws.disconnect()
            await self.kalshi_rest.close()

    async def _run_databento_collector(self, symbols: list[str]):
        """Run Databento collector."""
        try:
            # Register callback
            self.databento.on_tick(self._on_databento_tick)

            # Connect to live feed
            await self.databento.connect_live(symbols)
            logger.info(f"Databento: Connected for {symbols}")

            # Listen for ticks
            async for tick in self.databento.listen():
                if not self._running:
                    break

        except Exception as e:
            logger.error(f"Databento collector error: {e}")
        finally:
            await self.databento.disconnect()

    async def _status_reporter(self, interval: float = 5.0):
        """Periodically report collection status."""
        while self._running:
            await asyncio.sleep(interval)
            self._log_status()

    def _log_status(self):
        """Log current collection status."""
        if not self._stats["start_time"]:
            return

        duration = (datetime.utcnow() - self._stats["start_time"]).total_seconds()
        disk_usage = self._get_disk_usage()

        logger.info(
            f"Status: {duration/60:.1f}m | "
            f"Kalshi: {self._stats['kalshi_orderbooks']} ob, {self._stats['kalshi_trades']} trades | "
            f"Databento: {self._stats['databento_ticks']} ticks | "
            f"ES: {self._stats['last_es_price']} | "
            f"Disk: {disk_usage:.1f} MB"
        )

    def _get_disk_usage(self) -> float:
        """Get disk usage of data directory in MB."""
        total = 0
        for f in self.data_dir.glob("*.parquet"):
            total += f.stat().st_size
        return total / (1024 * 1024)

    def get_status_table(self) -> Table:
        """Create rich table with current status."""
        table = Table(title="Data Collection Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        if self._stats["start_time"]:
            duration = (datetime.utcnow() - self._stats["start_time"]).total_seconds()
            table.add_row("Duration", f"{duration/60:.1f} minutes")

        table.add_row("Kalshi Orderbooks", str(self._stats["kalshi_orderbooks"]))
        table.add_row("Kalshi Trades", str(self._stats["kalshi_trades"]))
        table.add_row("Databento Ticks", str(self._stats["databento_ticks"]))
        table.add_row("ES Price", str(self._stats["last_es_price"] or "N/A"))
        table.add_row("Disk Usage", f"{self._get_disk_usage():.1f} MB")

        return table

    async def start(
        self,
        kalshi_series: str = "INXD",
        databento_symbols: list[str] = None,
    ):
        """
        Start both collectors.

        Args:
            kalshi_series: Kalshi series to collect (e.g., "INXD")
            databento_symbols: Databento symbols (default: ["ES.FUT"])
        """
        if databento_symbols is None:
            databento_symbols = ["ES.FUT"]

        self._running = True
        self._stats["start_time"] = datetime.utcnow()

        # Discover Kalshi markets
        kalshi_tickers = await self.discover_kalshi_markets(kalshi_series)

        if not kalshi_tickers:
            logger.warning("No Kalshi markets found")
            return

        # Run both collectors concurrently
        await asyncio.gather(
            self._run_kalshi_collector(kalshi_tickers),
            self._run_databento_collector(databento_symbols),
            self._status_reporter(),
            return_exceptions=True,
        )

    async def stop(self):
        """Stop collectors and flush data."""
        logger.info("Stopping collectors...")
        self._running = False

        # Give collectors time to clean up
        await asyncio.sleep(1)

        # Flush all buffered data
        self.recorder.flush_all()
        logger.info("Data flushed to disk")

        # Final status
        self._log_status()


@click.command()
@click.option(
    "--markets",
    "-m",
    default="INXD",
    help="Kalshi series ticker (e.g., INXD for S&P 500 index)",
)
@click.option(
    "--env",
    "-e",
    type=click.Choice(["demo", "prod"]),
    default="demo",
    help="Kalshi environment",
)
@click.option(
    "--data-dir",
    "-d",
    default="data/raw",
    help="Directory to store collected data",
)
@click.option(
    "--es-symbol",
    "-s",
    default="ES.FUT",
    help="ES futures symbol (ES.FUT for continuous)",
)
def main(markets: str, env: str, data_dir: str, es_symbol: str):
    """
    Run combined Kalshi and Databento data collector.

    Collects both Kalshi prediction market data and ES futures data
    concurrently for arbitrage backtesting.

    IMPORTANT: Run this for 1-2 weeks before backtesting to build
    sufficient historical data.

    Examples:
        # Start collecting INXD markets and ES futures
        python scripts/run_collector.py --markets INXD --env demo

        # Collect production data
        python scripts/run_collector.py --markets INXD --env prod
    """
    # Get credentials from environment
    kalshi_api_key = os.getenv("KALSHI_API_KEY")
    kalshi_api_secret = os.getenv("KALSHI_API_SECRET")
    databento_api_key = os.getenv("DATABENTO_API_KEY")

    missing = []
    if not kalshi_api_key:
        missing.append("KALSHI_API_KEY")
    if not kalshi_api_secret:
        missing.append("KALSHI_API_SECRET")
    if not databento_api_key:
        missing.append("DATABENTO_API_KEY")

    if missing:
        raise click.ClickException(
            f"Missing environment variables: {', '.join(missing)}\n"
            "Set these in a .env file or export them."
        )

    collector = CombinedCollector(
        kalshi_api_key=kalshi_api_key,
        kalshi_api_secret=kalshi_api_secret,
        databento_api_key=databento_api_key,
        kalshi_env=env,
        data_dir=data_dir,
    )

    async def run():
        # Setup graceful shutdown
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def handle_shutdown():
            console.print("\n[yellow]Shutdown signal received...[/yellow]")
            asyncio.create_task(collector.stop())
            shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_shutdown)

        console.print(f"[green]Starting data collection[/green]")
        console.print(f"  Kalshi: {markets} ({env})")
        console.print(f"  Databento: {es_symbol}")
        console.print(f"  Output: {data_dir}")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        # Start collectors
        await collector.start(
            kalshi_series=markets,
            databento_symbols=[es_symbol],
        )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass

    console.print("[green]Collection complete![/green]")


if __name__ == "__main__":
    main()
