"""
Databento ES futures data collector - records market data to Parquet.

This script streams live ES (E-mini S&P 500) futures data from Databento:
- Best bid/ask prices and sizes (L1)
- Market-by-price orderbook (L2, up to 10 levels)

The underlying ES price is used to calculate fair value for Kalshi binary options.

Usage:
    python scripts/collect_databento.py --symbol ES.FUT
"""

import asyncio
import logging
import os
import signal
from datetime import datetime
from typing import Optional

import click
from dotenv import load_dotenv

from src.data_feed.databento_client import DatabentoClient
from src.data_feed.recorder import DataRecorder
from src.data_feed.schemas import UnderlyingTick

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DatabentoCollector:
    """
    Collects ES futures market data from Databento.

    Streams live MBP-10 (market-by-price, 10 levels) data and records
    tick data to Parquet files for backtesting.
    """

    def __init__(
        self,
        api_key: str,
        data_dir: str = "data/raw",
    ):
        self.client = DatabentoClient(api_key)
        self.recorder = DataRecorder(data_dir)
        self._running = False
        self._stats = {
            "ticks": 0,
            "start_time": None,
            "last_price": None,
        }

    async def start(self, symbols: list[str]):
        """
        Start collecting data for specified symbols.

        Args:
            symbols: List of symbols to subscribe to (e.g., ["ES.FUT"])
        """
        self._running = True
        self._stats["start_time"] = datetime.utcnow()

        # Connect to live feed
        await self.client.connect_live(symbols)
        logger.info(f"Connected to Databento live feed for {symbols}")

        # Listen for ticks
        try:
            async for tick in self.client.listen():
                if not self._running:
                    break

                self.recorder.record_underlying_tick(tick)
                self._stats["ticks"] += 1
                self._stats["last_price"] = tick.price

                # Log progress periodically
                if self._stats["ticks"] % 10000 == 0:
                    self._log_progress()

        except Exception as e:
            logger.error(f"Databento error: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop collecting and flush data."""
        self._running = False
        await self.client.disconnect()
        self.recorder.flush_all()

        # Log final stats
        self._log_progress(final=True)

    def _log_progress(self, final: bool = False):
        """Log collection progress."""
        if self._stats["start_time"]:
            duration = (datetime.utcnow() - self._stats["start_time"]).total_seconds()
            ticks_per_sec = (
                self._stats["ticks"] / duration if duration > 0 else 0
            )
            status = "Final" if final else "Progress"
            logger.info(
                f"{status}: {self._stats['ticks']} ticks, "
                f"{ticks_per_sec:.1f} ticks/sec, "
                f"last price: {self._stats['last_price']}"
            )

    def get_stats(self) -> dict:
        """Get collection statistics."""
        stats = self._stats.copy()
        if stats["start_time"]:
            stats["duration_seconds"] = (
                datetime.utcnow() - stats["start_time"]
            ).total_seconds()
            if stats["duration_seconds"] > 0:
                stats["ticks_per_sec"] = (
                    stats["ticks"] / stats["duration_seconds"]
                )
        return stats


@click.command()
@click.option(
    "--symbol",
    "-s",
    default="ES.FUT",
    help="Symbol to collect (ES.FUT for continuous front month)",
)
@click.option(
    "--data-dir",
    "-d",
    default="data/raw",
    help="Directory to store collected data",
)
def main(symbol: str, data_dir: str):
    """
    Collect ES futures market data from Databento.

    Streams live market data and records to Parquet files for backtesting.
    Uses MBP-10 schema (market-by-price with 10 levels of depth).

    Examples:
        # Collect ES continuous front month
        python scripts/collect_databento.py --symbol ES.FUT

        # Collect specific contract
        python scripts/collect_databento.py --symbol ESH5
    """
    # Get API key from environment
    api_key = os.getenv("DATABENTO_API_KEY")

    if not api_key:
        raise click.ClickException(
            "DATABENTO_API_KEY environment variable required"
        )

    collector = DatabentoCollector(
        api_key=api_key,
        data_dir=data_dir,
    )

    async def run():
        symbols = [symbol]

        # Setup graceful shutdown
        loop = asyncio.get_running_loop()

        def handle_shutdown():
            logger.info("Shutdown signal received...")
            asyncio.create_task(collector.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_shutdown)

        # Start collecting
        await collector.start(symbols)

    asyncio.run(run())


if __name__ == "__main__":
    main()
