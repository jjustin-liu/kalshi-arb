"""
Kalshi data collector - records orderbooks and trades to Parquet.

This script connects to Kalshi's WebSocket feed and records:
- Orderbook snapshots/deltas (bid/ask levels, depths, spreads)
- Trade executions (price, size, taker side)

Usage:
    python scripts/collect_kalshi.py --markets INXD --env demo
"""

import asyncio
import logging
import os
import signal
from datetime import datetime
from typing import Optional

import click
from dotenv import load_dotenv

from src.data_feed.kalshi_client import KalshiRESTClient, KalshiWebSocketClient
from src.data_feed.recorder import DataRecorder
from src.data_feed.schemas import KalshiOrderbook, KalshiTrade

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class KalshiCollector:
    """
    Collects Kalshi market data via WebSocket.

    Records orderbook updates and trades to Parquet files using DataRecorder.
    Automatically discovers active markets in specified event series.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        env: str = "demo",
        data_dir: str = "data/raw",
    ):
        self.rest_client = KalshiRESTClient(api_key, api_secret, env)
        self.ws_client = KalshiWebSocketClient(api_key, api_secret, env)
        self.recorder = DataRecorder(data_dir)
        self._running = False
        self._stats = {
            "orderbook_updates": 0,
            "trades": 0,
            "start_time": None,
        }

    async def discover_markets(self, series_ticker: str) -> list[str]:
        """
        Discover active markets for an event series.

        Args:
            series_ticker: Series ticker like "INXD" (S&P 500 index daily)

        Returns:
            List of market tickers to subscribe to
        """
        markets = []
        cursor = None

        while True:
            batch, cursor = await self.rest_client.get_markets(
                series_ticker=series_ticker,
                status="open",
                cursor=cursor,
            )
            markets.extend(batch)

            if not cursor:
                break

        tickers = [m.ticker for m in markets]
        logger.info(f"Discovered {len(tickers)} active markets for {series_ticker}")
        return tickers

    def _on_orderbook(self, orderbook: KalshiOrderbook):
        """Handle orderbook update."""
        self.recorder.record_kalshi_orderbook(orderbook)
        self._stats["orderbook_updates"] += 1

    def _on_trade(self, trade: KalshiTrade):
        """Handle trade update."""
        self.recorder.record_kalshi_trade(trade)
        self._stats["trades"] += 1

    async def start(self, market_tickers: list[str]):
        """
        Start collecting data for specified markets.

        Args:
            market_tickers: List of market tickers to subscribe to
        """
        self._running = True
        self._stats["start_time"] = datetime.utcnow()

        # Register callbacks
        self.ws_client.on_orderbook(self._on_orderbook)
        self.ws_client.on_trade(self._on_trade)

        # Connect WebSocket
        await self.ws_client.connect()

        # Subscribe to each market
        for ticker in market_tickers:
            await self.ws_client.subscribe_orderbook(ticker)
            await self.ws_client.subscribe_trades(ticker)
            await asyncio.sleep(0.1)  # Rate limit subscriptions

        logger.info(f"Subscribed to {len(market_tickers)} markets")

        # Listen for messages
        try:
            async for _ in self.ws_client.listen():
                if not self._running:
                    break
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop collecting and flush data."""
        self._running = False
        await self.ws_client.disconnect()
        await self.rest_client.close()
        self.recorder.flush_all()

        # Log final stats
        if self._stats["start_time"]:
            duration = (datetime.utcnow() - self._stats["start_time"]).total_seconds()
            logger.info(
                f"Collection stopped. Duration: {duration:.0f}s, "
                f"Orderbooks: {self._stats['orderbook_updates']}, "
                f"Trades: {self._stats['trades']}"
            )

    def get_stats(self) -> dict:
        """Get collection statistics."""
        stats = self._stats.copy()
        if stats["start_time"]:
            stats["duration_seconds"] = (
                datetime.utcnow() - stats["start_time"]
            ).total_seconds()
            if stats["duration_seconds"] > 0:
                stats["orderbooks_per_sec"] = (
                    stats["orderbook_updates"] / stats["duration_seconds"]
                )
                stats["trades_per_sec"] = (
                    stats["trades"] / stats["duration_seconds"]
                )
        return stats


@click.command()
@click.option(
    "--markets",
    "-m",
    default="INXD",
    help="Series ticker to collect (e.g., INXD for S&P 500 index)",
)
@click.option(
    "--env",
    "-e",
    type=click.Choice(["demo", "prod"]),
    default="demo",
    help="Kalshi environment (demo or prod)",
)
@click.option(
    "--data-dir",
    "-d",
    default="data/raw",
    help="Directory to store collected data",
)
@click.option(
    "--specific-tickers",
    "-t",
    multiple=True,
    help="Specific market tickers to collect (overrides --markets)",
)
def main(
    markets: str,
    env: str,
    data_dir: str,
    specific_tickers: tuple[str, ...],
):
    """
    Collect Kalshi orderbook and trade data.

    Connects to Kalshi WebSocket feed and records market data to Parquet files.
    Data is partitioned by date for efficient backtesting.

    Examples:
        # Collect all INXD markets (S&P 500 index)
        python scripts/collect_kalshi.py --markets INXD --env demo

        # Collect specific market tickers
        python scripts/collect_kalshi.py -t INXD-24JAN15-B5850 -t INXD-24JAN15-B5900
    """
    # Get API credentials from environment
    api_key = os.getenv("KALSHI_API_KEY")
    api_secret = os.getenv("KALSHI_API_SECRET")

    if not api_key or not api_secret:
        raise click.ClickException(
            "KALSHI_API_KEY and KALSHI_API_SECRET environment variables required"
        )

    collector = KalshiCollector(
        api_key=api_key,
        api_secret=api_secret,
        env=env,
        data_dir=data_dir,
    )

    async def run():
        # Get market tickers
        if specific_tickers:
            tickers = list(specific_tickers)
        else:
            tickers = await collector.discover_markets(markets)

        if not tickers:
            logger.warning("No markets found to collect")
            return

        # Setup graceful shutdown
        loop = asyncio.get_running_loop()

        def handle_shutdown():
            logger.info("Shutdown signal received...")
            asyncio.create_task(collector.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_shutdown)

        # Start collecting
        await collector.start(tickers)

    asyncio.run(run())


if __name__ == "__main__":
    main()
