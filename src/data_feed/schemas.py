"""Data schemas for the arbitrage system."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class Side(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class MarketStatus(Enum):
    """Kalshi market status."""
    ACTIVE = "active"
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"
    FINALIZED = "finalized"
    PENDING = "pending"


@dataclass(frozen=True)
class PriceLevel:
    """Single price level in orderbook."""
    price: int  # Cents (1-99 for Kalshi)
    quantity: int  # Number of contracts


@dataclass
class KalshiOrderbook:
    """Kalshi market orderbook snapshot."""
    market_ticker: str
    timestamp: datetime
    yes_bids: list[PriceLevel] = field(default_factory=list)
    yes_asks: list[PriceLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[int]:
        """Best bid price for YES contracts."""
        return self.yes_bids[0].price if self.yes_bids else None

    @property
    def best_ask(self) -> Optional[int]:
        """Best ask price for YES contracts."""
        return self.yes_asks[0].price if self.yes_asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid price for YES contracts."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[int]:
        """Bid-ask spread in cents."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def implied_probability(self) -> Optional[float]:
        """Implied probability from mid price (0-1 scale)."""
        if self.mid_price:
            return self.mid_price / 100
        return None

    def depth_at_price(self, side: Side, price: int) -> int:
        """Get total quantity available at or better than price."""
        if side == Side.BUY:
            return sum(lvl.quantity for lvl in self.yes_asks if lvl.price <= price)
        else:
            return sum(lvl.quantity for lvl in self.yes_bids if lvl.price >= price)


@dataclass
class KalshiMarket:
    """Kalshi market metadata."""
    ticker: str
    event_ticker: str
    title: str
    strike_price: Optional[float]  # For index markets like INXD
    expiry: datetime
    status: MarketStatus
    result: Optional[str] = None  # "yes" or "no" if settled
    volume: int = 0
    open_interest: int = 0

    @property
    def time_to_expiry(self) -> float:
        """Time to expiry in years."""
        delta = self.expiry - datetime.utcnow()
        return max(0, delta.total_seconds() / (365.25 * 24 * 3600))


@dataclass
class KalshiTrade:
    """Kalshi trade execution."""
    market_ticker: str
    timestamp: datetime
    price: int  # Cents
    quantity: int
    taker_side: Side


@dataclass
class UnderlyingTick:
    """Tick data from underlying market (ES futures)."""
    symbol: str
    timestamp: datetime
    price: float  # ES futures price
    size: int

    # Order book data (optional, for L2)
    bid_price: Optional[float] = None
    bid_size: Optional[int] = None
    ask_price: Optional[float] = None
    ask_size: Optional[int] = None


@dataclass
class UnderlyingOrderbook:
    """L2 orderbook for underlying asset."""
    symbol: str
    timestamp: datetime
    bids: list[tuple[float, int]] = field(default_factory=list)  # (price, size)
    asks: list[tuple[float, int]] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


@dataclass
class ArbitrageSignal:
    """Signal generated when arbitrage opportunity detected."""
    timestamp: datetime
    market_ticker: str
    underlying_symbol: str

    # Prices
    underlying_price: float
    strike_price: float
    kalshi_mid: float  # 0-100

    # Probabilities
    fair_probability: float  # 0-1, from Black-Scholes
    implied_probability: float  # 0-1, from Kalshi mid

    # Edge calculation
    basis: float  # fair - implied
    expected_fees: float
    expected_slippage: float
    net_edge: float  # basis - fees - slippage

    # Risk metrics
    toxicity_score: float  # 0-1
    volatility: float
    time_to_expiry: float  # Years

    # Recommendation
    side: Side  # BUY if fair > implied, SELL otherwise
    recommended_size: int
    confidence: float  # 0-1

    @property
    def is_tradeable(self) -> bool:
        """Check if signal meets trading criteria."""
        return (
            self.net_edge > 0.01 and  # 100 bps minimum
            self.toxicity_score < 0.6 and
            self.recommended_size > 0
        )


@dataclass
class Order:
    """Order to be submitted."""
    market_ticker: str
    side: Side
    price: int  # Limit price in cents
    quantity: int
    client_order_id: str

    # Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None


@dataclass
class Fill:
    """Trade fill/execution."""
    order_id: str
    market_ticker: str
    side: Side
    price: int
    quantity: int
    timestamp: datetime
    fee: float


@dataclass
class Position:
    """Current position in a market."""
    market_ticker: str
    quantity: int  # Positive = long YES, negative = short YES (long NO)
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class ToxicityMetrics:
    """Toxic flow detection metrics."""
    timestamp: datetime

    # Order Flow Imbalance
    ofi: float  # Signed OFI
    ofi_zscore: float  # Standardized OFI

    # VPIN
    vpin: float  # 0-1

    # Spread metrics
    spread: float
    spread_zscore: float

    # Sweep detection
    sweep_detected: bool
    levels_cleared: int

    # Combined score
    toxicity_score: float  # 0-1 combined score

    @property
    def is_toxic(self) -> bool:
        """Check if current conditions are toxic."""
        return self.toxicity_score > 0.6


@dataclass
class BacktestTrade:
    """Trade record for backtesting."""
    timestamp: datetime
    market_ticker: str
    side: Side
    entry_price: int
    exit_price: Optional[int]
    quantity: int

    # Attribution
    signal_edge: float  # Expected edge at signal time
    realized_edge: float  # Actual realized edge
    slippage: float  # Entry slippage
    fees: float

    # Outcome
    pnl: float
    holding_period: Optional[float] = None  # Seconds

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class PerformanceMetrics:
    """Backtest or live performance metrics."""
    start_date: datetime
    end_date: datetime

    # Returns
    total_pnl: float
    total_return: float  # As percentage
    annualized_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: float  # Days

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # Gross profit / gross loss

    # Execution quality
    fill_rate: float
    avg_slippage: float
    edge_captured: float  # Realized / expected edge

    # Toxicity
    toxicity_saves: int  # Trades avoided due to toxicity
    toxicity_precision: float  # Correctly avoided bad trades


@dataclass
class PnLAttribution:
    """PnL decomposition by source."""
    timestamp: datetime
    period: str  # "daily", "weekly", etc.

    # Components
    signal_pnl: float  # From correct directional calls
    execution_pnl: float  # From execution quality
    toxicity_saves: float  # Avoided losses from toxic filter
    fees_paid: float
    slippage_cost: float

    @property
    def total_pnl(self) -> float:
        return (
            self.signal_pnl +
            self.execution_pnl +
            self.toxicity_saves -
            self.fees_paid -
            self.slippage_cost
        )
