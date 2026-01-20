"""
Dashboard State Management.

Maintains current state of the trading system for display in the web UI.
Provides methods to update state and serialize for JSON API/WebSocket.
"""

from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any
import json


@dataclass
class MarketData:
    """Current state of a single market."""
    ticker: str
    best_bid: Optional[int] = None
    best_ask: Optional[int] = None
    spread: Optional[int] = None
    mid_price: Optional[float] = None
    bid_depth: int = 0
    ask_depth: int = 0
    last_trade_price: Optional[int] = None
    last_trade_time: Optional[datetime] = None
    volume: int = 0
    toxicity_score: float = 0.0
    ofi: float = 0.0
    vpin: float = 0.0


@dataclass
class PositionData:
    """Current position in a market."""
    market: str
    quantity: int
    avg_entry: float
    unrealized_pnl: float
    realized_pnl: float
    last_update: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SignalData:
    """Recent signal data."""
    timestamp: datetime
    market: str
    side: str
    fair_prob: float
    implied_prob: float
    basis: float
    net_edge: float
    toxicity: float
    recommended_size: int
    tradeable: bool
    traded: bool = False


@dataclass
class TradeData:
    """Executed trade data."""
    timestamp: datetime
    market: str
    side: str
    price: int
    quantity: int
    fees: float
    pnl: float


@dataclass
class DashboardState:
    """
    Complete dashboard state.

    Contains all data needed to render the monitoring dashboard.
    """

    # Timestamps
    last_update: datetime = field(default_factory=datetime.utcnow)
    system_start: datetime = field(default_factory=datetime.utcnow)

    # System status
    is_running: bool = False
    is_trading_enabled: bool = False
    error_message: Optional[str] = None

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    drawdown_pct: float = 0.0

    # Underlying
    es_price: Optional[float] = None
    es_bid: Optional[float] = None
    es_ask: Optional[float] = None

    # Positions
    total_position: int = 0
    positions: dict[str, PositionData] = field(default_factory=dict)

    # Markets
    markets: dict[str, MarketData] = field(default_factory=dict)

    # Hedging
    hedge_position: float = 0.0
    hedge_pnl: float = 0.0
    net_delta: float = 0.0

    # Statistics
    signals_generated: int = 0
    signals_traded: int = 0
    signals_filtered: int = 0
    total_trades: int = 0
    total_volume: int = 0

    # Recent history (as lists for JSON serialization)
    recent_signals: list = field(default_factory=list)
    recent_trades: list = field(default_factory=list)


class DashboardManager:
    """
    Manages dashboard state and provides update methods.

    Thread-safe state management for the monitoring dashboard.

    Usage:
        manager = DashboardManager()

        # Update state
        manager.update_market(market_data)
        manager.update_position(market, quantity, avg_entry)
        manager.add_signal(signal_data)
        manager.add_trade(trade_data)

        # Get state for API
        state = manager.get_state()
        json_state = manager.to_json()
    """

    def __init__(
        self,
        max_signals: int = 100,
        max_trades: int = 100,
        initial_capital: float = 10000.0,
    ):
        """
        Initialize dashboard manager.

        Args:
            max_signals: Maximum signals to keep in history
            max_trades: Maximum trades to keep in history
            initial_capital: Starting capital
        """
        self._state = DashboardState(
            peak_equity=initial_capital,
            current_equity=initial_capital,
        )
        self._signals_queue = deque(maxlen=max_signals)
        self._trades_queue = deque(maxlen=max_trades)
        self._callbacks: list = []

    def on_update(self, callback):
        """Register callback for state updates."""
        self._callbacks.append(callback)

    def _notify(self):
        """Notify callbacks of state update."""
        self._state.last_update = datetime.utcnow()
        for callback in self._callbacks:
            try:
                callback(self._state)
            except Exception:
                pass

    def update_system_status(
        self,
        is_running: bool,
        is_trading_enabled: bool,
        error: Optional[str] = None,
    ):
        """Update system status."""
        self._state.is_running = is_running
        self._state.is_trading_enabled = is_trading_enabled
        self._state.error_message = error
        self._notify()

    def update_pnl(
        self,
        realized: float,
        unrealized: float,
        daily: float,
        equity: float,
        peak: float,
    ):
        """Update P&L metrics."""
        self._state.realized_pnl = realized
        self._state.unrealized_pnl = unrealized
        self._state.total_pnl = realized + unrealized
        self._state.daily_pnl = daily
        self._state.current_equity = equity
        self._state.peak_equity = max(peak, self._state.peak_equity)
        self._state.drawdown_pct = (
            (self._state.peak_equity - equity) / self._state.peak_equity
            if self._state.peak_equity > 0 else 0
        )
        self._notify()

    def update_underlying(
        self,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ):
        """Update underlying (ES) price."""
        self._state.es_price = price
        self._state.es_bid = bid
        self._state.es_ask = ask
        self._notify()

    def update_market(self, market: MarketData):
        """Update market data."""
        self._state.markets[market.ticker] = market
        self._notify()

    def update_position(
        self,
        market: str,
        quantity: int,
        avg_entry: float,
        unrealized_pnl: float,
        realized_pnl: float,
    ):
        """Update position for a market."""
        self._state.positions[market] = PositionData(
            market=market,
            quantity=quantity,
            avg_entry=avg_entry,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
        )
        self._state.total_position = sum(
            abs(p.quantity) for p in self._state.positions.values()
        )
        self._notify()

    def update_hedge(
        self,
        position: float,
        pnl: float,
        net_delta: float,
    ):
        """Update hedge data."""
        self._state.hedge_position = position
        self._state.hedge_pnl = pnl
        self._state.net_delta = net_delta
        self._notify()

    def add_signal(self, signal: SignalData):
        """Add a signal to history."""
        self._signals_queue.append(signal)
        self._state.signals_generated += 1
        if signal.traded:
            self._state.signals_traded += 1
        elif signal.tradeable:
            self._state.signals_filtered += 1
        self._state.recent_signals = list(self._signals_queue)
        self._notify()

    def add_trade(self, trade: TradeData):
        """Add a trade to history."""
        self._trades_queue.append(trade)
        self._state.total_trades += 1
        self._state.total_volume += trade.quantity
        self._state.recent_trades = list(self._trades_queue)
        self._notify()

    def update_stats(
        self,
        signals_generated: int,
        signals_traded: int,
        signals_filtered: int,
        total_trades: int,
        total_volume: int,
    ):
        """Update summary statistics."""
        self._state.signals_generated = signals_generated
        self._state.signals_traded = signals_traded
        self._state.signals_filtered = signals_filtered
        self._state.total_trades = total_trades
        self._state.total_volume = total_volume
        self._notify()

    def get_state(self) -> DashboardState:
        """Get current state."""
        return self._state

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        state = self._state

        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        def serialize_dataclass(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: serialize_datetime(v) for k, v in asdict(obj).items()}
            return obj

        return {
            "last_update": state.last_update.isoformat(),
            "system_start": state.system_start.isoformat(),
            "is_running": state.is_running,
            "is_trading_enabled": state.is_trading_enabled,
            "error_message": state.error_message,
            "pnl": {
                "realized": state.realized_pnl,
                "unrealized": state.unrealized_pnl,
                "total": state.total_pnl,
                "daily": state.daily_pnl,
                "peak_equity": state.peak_equity,
                "current_equity": state.current_equity,
                "drawdown_pct": state.drawdown_pct,
            },
            "underlying": {
                "price": state.es_price,
                "bid": state.es_bid,
                "ask": state.es_ask,
            },
            "positions": {
                k: serialize_dataclass(v)
                for k, v in state.positions.items()
            },
            "markets": {
                k: serialize_dataclass(v)
                for k, v in state.markets.items()
            },
            "hedge": {
                "position": state.hedge_position,
                "pnl": state.hedge_pnl,
                "net_delta": state.net_delta,
            },
            "stats": {
                "signals_generated": state.signals_generated,
                "signals_traded": state.signals_traded,
                "signals_filtered": state.signals_filtered,
                "total_trades": state.total_trades,
                "total_volume": state.total_volume,
                "total_position": state.total_position,
            },
            "recent_signals": [
                serialize_dataclass(s) for s in state.recent_signals[-20:]
            ],
            "recent_trades": [
                serialize_dataclass(t) for t in state.recent_trades[-20:]
            ],
        }

    def to_json(self) -> str:
        """Convert state to JSON string."""
        return json.dumps(self.to_dict())
