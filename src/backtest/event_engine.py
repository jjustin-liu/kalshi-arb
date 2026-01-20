"""
Event-Driven Backtest Engine.

Simulates market events in chronological order, allowing strategies
to react to orderbook updates, trades, and underlying price changes.

Event types:
- orderbook: Kalshi orderbook update
- trade: Kalshi market trade
- underlying: ES futures tick
- signal: Generated arbitrage signal
- fill: Simulated fill
"""

import heapq
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Iterator, Optional

import pandas as pd


class EventType(Enum):
    """Types of backtest events."""
    ORDERBOOK = "orderbook"
    TRADE = "trade"
    UNDERLYING = "underlying"
    SIGNAL = "signal"
    FILL = "fill"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"


@dataclass(order=True)
class Event:
    """
    Single event in the backtest timeline.

    Events are ordered by timestamp, then by priority (lower = earlier).
    """
    timestamp: datetime
    priority: int = field(compare=True, default=0)
    event_type: EventType = field(compare=False, default=EventType.ORDERBOOK)
    data: Any = field(compare=False, default=None)
    market: str = field(compare=False, default="")

    def __post_init__(self):
        # Set priority based on type (ensures consistent ordering)
        type_priorities = {
            EventType.MARKET_OPEN: 0,
            EventType.UNDERLYING: 1,
            EventType.ORDERBOOK: 2,
            EventType.TRADE: 3,
            EventType.SIGNAL: 4,
            EventType.FILL: 5,
            EventType.MARKET_CLOSE: 6,
        }
        if self.priority == 0:
            self.priority = type_priorities.get(self.event_type, 5)


class EventEngine:
    """
    Event-driven simulation engine.

    Maintains a priority queue of events and dispatches them
    to registered handlers in chronological order.

    Usage:
        engine = EventEngine()

        # Register handlers
        engine.on(EventType.ORDERBOOK, handle_orderbook)
        engine.on(EventType.TRADE, handle_trade)

        # Load events
        engine.load_orderbooks(orderbook_df)
        engine.load_trades(trade_df)

        # Run simulation
        for event in engine.run():
            # Process event
            pass
    """

    def __init__(self):
        self._event_queue: list[Event] = []
        self._handlers: dict[EventType, list[Callable[[Event], None]]] = {
            et: [] for et in EventType
        }
        self._current_time: Optional[datetime] = None
        self._events_processed: int = 0

    def on(self, event_type: EventType, handler: Callable[[Event], None]):
        """
        Register handler for event type.

        Args:
            event_type: Type of event to handle
            handler: Callback function receiving Event
        """
        self._handlers[event_type].append(handler)

    def add_event(self, event: Event):
        """Add event to queue."""
        heapq.heappush(self._event_queue, event)

    def add_events(self, events: list[Event]):
        """Add multiple events to queue."""
        for event in events:
            self.add_event(event)

    def load_orderbooks(self, df: pd.DataFrame, market_col: str = "market_ticker"):
        """
        Load orderbook data into event queue.

        Args:
            df: DataFrame with orderbook snapshots
            market_col: Column containing market ticker
        """
        from src.data_feed.schemas import KalshiOrderbook, PriceLevel

        for _, row in df.iterrows():
            # Reconstruct orderbook from recorded data
            # Note: This is simplified - full reconstruction would need all levels
            orderbook = KalshiOrderbook(
                market_ticker=row.get(market_col, ""),
                timestamp=row["timestamp"],
                yes_bids=[PriceLevel(row.get("best_bid", 0), row.get("bid_depth_1", 0))]
                if row.get("best_bid") else [],
                yes_asks=[PriceLevel(row.get("best_ask", 0), row.get("ask_depth_1", 0))]
                if row.get("best_ask") else [],
            )

            self.add_event(Event(
                timestamp=row["timestamp"],
                event_type=EventType.ORDERBOOK,
                data=orderbook,
                market=row.get(market_col, ""),
            ))

    def load_trades(self, df: pd.DataFrame, market_col: str = "market_ticker"):
        """
        Load trade data into event queue.

        Args:
            df: DataFrame with trades
            market_col: Column containing market ticker
        """
        from src.data_feed.schemas import KalshiTrade, Side

        for _, row in df.iterrows():
            trade = KalshiTrade(
                market_ticker=row.get(market_col, ""),
                timestamp=row["timestamp"],
                price=row["price"],
                quantity=row["quantity"],
                taker_side=Side.BUY if row.get("side") == "buy" else Side.SELL,
            )

            self.add_event(Event(
                timestamp=row["timestamp"],
                event_type=EventType.TRADE,
                data=trade,
                market=row.get(market_col, ""),
            ))

    def load_underlying(self, df: pd.DataFrame, symbol: str = "ES.FUT"):
        """
        Load underlying tick data into event queue.

        Args:
            df: DataFrame with underlying ticks
            symbol: Underlying symbol
        """
        from src.data_feed.schemas import UnderlyingTick

        for _, row in df.iterrows():
            tick = UnderlyingTick(
                symbol=symbol,
                timestamp=row["timestamp"],
                price=row["price"],
                size=row.get("size", 0),
                bid_price=row.get("bid_price"),
                bid_size=row.get("bid_size"),
                ask_price=row.get("ask_price"),
                ask_size=row.get("ask_size"),
            )

            self.add_event(Event(
                timestamp=row["timestamp"],
                event_type=EventType.UNDERLYING,
                data=tick,
                market=symbol,
            ))

    def run(self) -> Iterator[Event]:
        """
        Run simulation, yielding events in chronological order.

        Yields:
            Events in timestamp order
        """
        while self._event_queue:
            event = heapq.heappop(self._event_queue)
            self._current_time = event.timestamp
            self._events_processed += 1

            # Call registered handlers
            for handler in self._handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    import logging
                    logging.error(f"Event handler error: {e}")

            yield event

    def run_until(self, end_time: datetime) -> Iterator[Event]:
        """
        Run simulation until specified time.

        Args:
            end_time: Stop when event timestamp exceeds this

        Yields:
            Events up to end_time
        """
        for event in self.run():
            if event.timestamp > end_time:
                # Put event back for later
                self.add_event(event)
                break
            yield event

    @property
    def current_time(self) -> Optional[datetime]:
        """Get current simulation time."""
        return self._current_time

    @property
    def events_remaining(self) -> int:
        """Get number of events remaining in queue."""
        return len(self._event_queue)

    @property
    def events_processed(self) -> int:
        """Get number of events processed."""
        return self._events_processed

    def peek(self) -> Optional[Event]:
        """Peek at next event without removing it."""
        return self._event_queue[0] if self._event_queue else None

    def clear(self):
        """Clear all events."""
        self._event_queue.clear()
        self._events_processed = 0
        self._current_time = None


class EventMerger:
    """
    Merges multiple event streams while maintaining chronological order.

    Useful for combining Kalshi and underlying data from different sources.
    """

    def __init__(self):
        self._streams: list[Iterator[Event]] = []
        self._current_events: list[tuple[Event, int]] = []

    def add_stream(self, events: Iterator[Event]):
        """Add event stream to merge."""
        stream_id = len(self._streams)
        self._streams.append(events)

        # Get first event from stream
        try:
            event = next(events)
            heapq.heappush(self._current_events, (event, stream_id))
        except StopIteration:
            pass

    def merge(self) -> Iterator[Event]:
        """
        Merge all streams in chronological order.

        Yields:
            Events from all streams in timestamp order
        """
        while self._current_events:
            # Get earliest event
            event, stream_id = heapq.heappop(self._current_events)
            yield event

            # Get next event from same stream
            try:
                next_event = next(self._streams[stream_id])
                heapq.heappush(self._current_events, (next_event, stream_id))
            except StopIteration:
                pass
