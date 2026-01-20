"""Databento client for underlying market data (ES futures)."""

import asyncio
import logging
import os
from datetime import datetime, date
from typing import AsyncIterator, Callable, Optional

import databento as db
from databento_dbn import UNDEF_PRICE

from .schemas import UnderlyingOrderbook, UnderlyingTick

logger = logging.getLogger(__name__)


# ES futures front month symbol pattern
ES_SYMBOL = "ES.FUT"  # Continuous front month
ES_DATASET = "GLBX.MDP3"  # CME Globex


class DatabentoClient:
    """Client for Databento market data feeds."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Databento API key required. Set DATABENTO_API_KEY "
                "environment variable or pass directly."
            )
        self._live_client: Optional[db.Live] = None
        self._historical_client: Optional[db.Historical] = None
        self._callbacks: dict[str, list[Callable]] = {
            "tick": [],
            "orderbook": [],
        }
        self._running = False

    def on_tick(self, callback: Callable[[UnderlyingTick], None]):
        """Register callback for tick updates."""
        self._callbacks["tick"].append(callback)

    def on_orderbook(self, callback: Callable[[UnderlyingOrderbook], None]):
        """Register callback for orderbook updates."""
        self._callbacks["orderbook"].append(callback)

    async def connect_live(self, symbols: Optional[list[str]] = None):
        """Connect to live data feed."""
        if symbols is None:
            symbols = [ES_SYMBOL]

        self._live_client = db.Live(key=self.api_key)
        self._live_client.subscribe(
            dataset=ES_DATASET,
            schema="mbp-10",  # Market-by-price, 10 levels
            symbols=symbols,
        )
        self._running = True
        logger.info(f"Connected to Databento live feed for {symbols}")

    async def disconnect(self):
        """Disconnect from live feed."""
        self._running = False
        if self._live_client:
            self._live_client.stop()
            self._live_client = None

    async def listen(self) -> AsyncIterator[UnderlyingTick]:
        """Listen for live market data."""
        if not self._live_client:
            raise RuntimeError("Not connected to live feed")

        for record in self._live_client:
            if not self._running:
                break

            tick = self._parse_record(record)
            if tick:
                yield tick
                for callback in self._callbacks["tick"]:
                    callback(tick)

    def _parse_record(self, record) -> Optional[UnderlyingTick]:
        """Parse Databento record to UnderlyingTick."""
        # Handle MBP (market-by-price) records
        if hasattr(record, "levels"):
            # Extract best bid/ask from levels
            bid_price = None
            bid_size = None
            ask_price = None
            ask_size = None

            if record.levels and len(record.levels) > 0:
                for level in record.levels:
                    if level.bid_px != UNDEF_PRICE and bid_price is None:
                        bid_price = level.bid_px / 1e9  # Convert fixed-point
                        bid_size = level.bid_sz
                    if level.ask_px != UNDEF_PRICE and ask_price is None:
                        ask_price = level.ask_px / 1e9
                        ask_size = level.ask_sz

            # Calculate mid price
            price = None
            if bid_price and ask_price:
                price = (bid_price + ask_price) / 2
            elif hasattr(record, "price") and record.price != UNDEF_PRICE:
                price = record.price / 1e9

            if price is None:
                return None

            return UnderlyingTick(
                symbol=record.hd.instrument_id if hasattr(record, "hd") else ES_SYMBOL,
                timestamp=datetime.fromtimestamp(record.ts_event / 1e9),
                price=price,
                size=record.size if hasattr(record, "size") else 0,
                bid_price=bid_price,
                bid_size=bid_size,
                ask_price=ask_price,
                ask_size=ask_size,
            )

        # Handle trade records
        elif hasattr(record, "price") and record.price != UNDEF_PRICE:
            return UnderlyingTick(
                symbol=record.hd.instrument_id if hasattr(record, "hd") else ES_SYMBOL,
                timestamp=datetime.fromtimestamp(record.ts_event / 1e9),
                price=record.price / 1e9,
                size=record.size if hasattr(record, "size") else 0,
            )

        return None

    def get_historical_client(self) -> db.Historical:
        """Get historical data client."""
        if self._historical_client is None:
            self._historical_client = db.Historical(key=self.api_key)
        return self._historical_client

    async def get_historical_data(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        schema: str = "mbp-10",
    ) -> list[UnderlyingTick]:
        """Fetch historical market data."""
        client = self.get_historical_client()

        data = client.timeseries.get_range(
            dataset=ES_DATASET,
            symbols=symbols,
            schema=schema,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        ticks = []
        for record in data:
            tick = self._parse_record(record)
            if tick:
                ticks.append(tick)

        logger.info(f"Fetched {len(ticks)} historical ticks")
        return ticks

    async def get_historical_orderbook(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        depth: int = 10,
    ) -> list[UnderlyingOrderbook]:
        """Fetch historical orderbook snapshots."""
        client = self.get_historical_client()

        schema = f"mbp-{depth}"
        data = client.timeseries.get_range(
            dataset=ES_DATASET,
            symbols=symbols,
            schema=schema,
            start=start.isoformat(),
            end=end.isoformat(),
        )

        orderbooks = []
        for record in data:
            if hasattr(record, "levels"):
                bids = []
                asks = []

                for level in record.levels:
                    if level.bid_px != UNDEF_PRICE:
                        bids.append((level.bid_px / 1e9, level.bid_sz))
                    if level.ask_px != UNDEF_PRICE:
                        asks.append((level.ask_px / 1e9, level.ask_sz))

                orderbooks.append(
                    UnderlyingOrderbook(
                        symbol=record.hd.instrument_id if hasattr(record, "hd") else ES_SYMBOL,
                        timestamp=datetime.fromtimestamp(record.ts_event / 1e9),
                        bids=sorted(bids, key=lambda x: -x[0]),  # Descending
                        asks=sorted(asks, key=lambda x: x[0]),  # Ascending
                    )
                )

        logger.info(f"Fetched {len(orderbooks)} historical orderbook snapshots")
        return orderbooks

    async def download_to_file(
        self,
        symbols: list[str],
        start: date,
        end: date,
        output_path: str,
        schema: str = "mbp-10",
    ):
        """Download historical data to DBN file."""
        client = self.get_historical_client()

        client.timeseries.get_range(
            dataset=ES_DATASET,
            symbols=symbols,
            schema=schema,
            start=start.isoformat(),
            end=end.isoformat(),
            path=output_path,
        )
        logger.info(f"Downloaded data to {output_path}")


class DatabentoReplayClient:
    """Client for replaying historical Databento data from files."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._store: Optional[db.DBNStore] = None

    def load(self):
        """Load DBN file."""
        self._store = db.DBNStore.from_file(self.file_path)
        logger.info(f"Loaded {self.file_path}")

    async def replay(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        speed: float = 1.0,
    ) -> AsyncIterator[UnderlyingTick]:
        """Replay historical data with optional time filtering and speed control."""
        if not self._store:
            self.load()

        last_ts = None
        for record in self._store:
            tick = self._parse_record(record)
            if tick is None:
                continue

            # Apply time filter
            if start and tick.timestamp < start:
                continue
            if end and tick.timestamp > end:
                break

            # Simulate real-time delay if speed > 0
            if speed > 0 and last_ts is not None:
                delay = (tick.timestamp - last_ts).total_seconds() / speed
                if delay > 0:
                    await asyncio.sleep(min(delay, 1.0))  # Cap at 1 second

            last_ts = tick.timestamp
            yield tick

    def _parse_record(self, record) -> Optional[UnderlyingTick]:
        """Parse record to UnderlyingTick."""
        if hasattr(record, "levels"):
            bid_price = None
            bid_size = None
            ask_price = None
            ask_size = None

            if record.levels:
                for level in record.levels:
                    if level.bid_px != UNDEF_PRICE and bid_price is None:
                        bid_price = level.bid_px / 1e9
                        bid_size = level.bid_sz
                    if level.ask_px != UNDEF_PRICE and ask_price is None:
                        ask_price = level.ask_px / 1e9
                        ask_size = level.ask_sz

            price = None
            if bid_price and ask_price:
                price = (bid_price + ask_price) / 2

            if price is None:
                return None

            return UnderlyingTick(
                symbol=str(record.hd.instrument_id) if hasattr(record, "hd") else ES_SYMBOL,
                timestamp=datetime.fromtimestamp(record.ts_event / 1e9),
                price=price,
                size=0,
                bid_price=bid_price,
                bid_size=bid_size,
                ask_price=ask_price,
                ask_size=ask_size,
            )

        return None
