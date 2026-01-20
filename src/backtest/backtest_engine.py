"""
Backtest Engine - Main orchestrator for historical simulation.

Coordinates:
- Data loading from Parquet files
- Event-driven simulation
- Signal generation
- Execution simulation
- Performance measurement
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Callable

import pandas as pd

from src.data_feed.recorder import DataLoader
from src.data_feed.schemas import (
    ArbitrageSignal,
    BacktestTrade,
    KalshiMarket,
    KalshiOrderbook,
    KalshiTrade,
    PerformanceMetrics,
    PnLAttribution,
    PriceLevel,
    Side,
    UnderlyingTick,
    MarketStatus,
)
from src.pricing.basis_calculator import BasisCalculator
from src.risk import RiskManager, RiskConfig

from .config import BacktestConfig
from .event_engine import EventEngine, Event, EventType
from .fill_model import BaseFillModel, create_fill_model
from .metrics import MetricsCalculator, EquityCurve

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    config: BacktestConfig
    metrics: PerformanceMetrics
    attribution: PnLAttribution
    equity_curve: EquityCurve
    trades: list[BacktestTrade]
    signals_generated: int
    signals_traded: int
    signals_filtered: int


@dataclass
class MarketState:
    """Current state of a market during backtest."""
    market: Optional[KalshiMarket] = None
    orderbook: Optional[KalshiOrderbook] = None
    last_trade: Optional[KalshiTrade] = None
    position: int = 0
    avg_entry: float = 0.0


class BacktestEngine:
    """
    Main backtesting engine.

    Simulates trading strategy on historical data to evaluate:
    - Strategy performance (Sharpe, returns)
    - Risk metrics (drawdown, volatility)
    - Execution quality (slippage, fill rate)

    Usage:
        engine = BacktestEngine(config)
        result = engine.run()
        print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        basis_calculator: Optional[BasisCalculator] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            basis_calculator: Signal generator (uses default if None)
            risk_manager: Risk checks (uses default if None)
        """
        self.config = config or BacktestConfig()

        # Components
        self.data_loader = DataLoader(self.config.data_dir)
        self.basis_calculator = basis_calculator or BasisCalculator(
            min_edge_threshold=self.config.min_edge_threshold,
        )
        self.risk_manager = risk_manager or RiskManager(
            config=RiskConfig(),
            initial_capital=self.config.initial_capital,
        )
        self.fill_model = create_fill_model(self.config.fill_model)
        self.metrics = MetricsCalculator(self.config.initial_capital)

        # Event engine
        self.event_engine = EventEngine()

        # State tracking
        self._market_states: dict[str, MarketState] = {}
        self._underlying: Optional[UnderlyingTick] = None
        self._trades: list[BacktestTrade] = []
        self._signals_generated = 0
        self._signals_traded = 0
        self._signals_filtered = 0

        # Callbacks
        self._on_signal: Optional[Callable[[ArbitrageSignal], None]] = None
        self._on_trade: Optional[Callable[[BacktestTrade], None]] = None

    def on_signal(self, callback: Callable[[ArbitrageSignal], None]):
        """Register callback for generated signals."""
        self._on_signal = callback

    def on_trade(self, callback: Callable[[BacktestTrade], None]):
        """Register callback for executed trades."""
        self._on_trade = callback

    def run(self) -> BacktestResult:
        """
        Run the backtest.

        Returns:
            BacktestResult with all performance data
        """
        logger.info(f"Starting backtest: {self.config.start_date} to {self.config.end_date}")

        # Load data
        self._load_data()

        # Register event handlers
        self.event_engine.on(EventType.ORDERBOOK, self._handle_orderbook)
        self.event_engine.on(EventType.TRADE, self._handle_trade)
        self.event_engine.on(EventType.UNDERLYING, self._handle_underlying)

        # Run simulation
        events_processed = 0
        for event in self.event_engine.run():
            events_processed += 1

            if events_processed % 100000 == 0:
                logger.info(f"Processed {events_processed} events...")

        logger.info(f"Backtest complete: {events_processed} events processed")

        # Close any open positions
        self._close_all_positions()

        # Calculate metrics
        metrics = self.metrics.calculate()
        attribution = self.metrics.get_attribution()
        equity_curve = self.metrics.get_equity_curve()

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            attribution=attribution,
            equity_curve=equity_curve,
            trades=self._trades,
            signals_generated=self._signals_generated,
            signals_traded=self._signals_traded,
            signals_filtered=self._signals_filtered,
        )

    def _load_data(self):
        """Load all data into event engine."""
        # Load Kalshi orderbooks
        orderbook_df = self.data_loader.load_kalshi_orderbook(
            self.config.start_date,
            self.config.end_date,
        )
        if not orderbook_df.empty:
            self.event_engine.load_orderbooks(orderbook_df)
            logger.info(f"Loaded {len(orderbook_df)} orderbook updates")

        # Load Kalshi trades
        trade_df = self.data_loader.load_kalshi_trades(
            self.config.start_date,
            self.config.end_date,
        )
        if not trade_df.empty:
            self.event_engine.load_trades(trade_df)
            logger.info(f"Loaded {len(trade_df)} trades")

        # Load underlying data
        underlying_df = self.data_loader.load_underlying_ticks(
            self.config.start_date,
            self.config.end_date,
        )
        if not underlying_df.empty:
            self.event_engine.load_underlying(underlying_df)
            logger.info(f"Loaded {len(underlying_df)} underlying ticks")

    def _handle_orderbook(self, event: Event):
        """Handle orderbook update event."""
        orderbook: KalshiOrderbook = event.data
        market = event.market

        # Update market state
        if market not in self._market_states:
            self._market_states[market] = MarketState()

        self._market_states[market].orderbook = orderbook

        # Update risk manager
        if self.config.toxicity_enabled:
            self.risk_manager.update_orderbook(orderbook)

        # Generate signal if we have underlying data
        if self._underlying:
            self._generate_signal(market)

    def _handle_trade(self, event: Event):
        """Handle Kalshi trade event."""
        trade: KalshiTrade = event.data
        market = event.market

        # Update market state
        if market not in self._market_states:
            self._market_states[market] = MarketState()

        self._market_states[market].last_trade = trade

        # Update risk manager
        if self.config.toxicity_enabled:
            self.risk_manager.update_trade(trade)

        # Process fills for queue-based model
        if hasattr(self.fill_model, 'on_trade'):
            fills = self.fill_model.on_trade(trade)
            for fill in fills:
                self._process_fill(fill, market)

    def _handle_underlying(self, event: Event):
        """Handle underlying tick event."""
        tick: UnderlyingTick = event.data
        self._underlying = tick

        # Update volatility manager
        self.basis_calculator.update_underlying(tick.timestamp, tick.price)

    def _generate_signal(self, market: str):
        """Generate trading signal for a market."""
        state = self._market_states.get(market)
        if not state or not state.orderbook:
            return

        # Create market object for pricing
        # Note: In production, this would come from API
        kalshi_market = self._create_market_object(market, state.orderbook)
        if not kalshi_market:
            return

        # Get toxicity
        toxicity = 0.0
        if self.config.toxicity_enabled:
            toxicity_metrics = self.risk_manager.get_toxicity_metrics(market, state.orderbook)
            toxicity = toxicity_metrics.toxicity_score

        # Generate signal
        signal = self.basis_calculator.calculate_signal(
            market=kalshi_market,
            orderbook=state.orderbook,
            underlying=self._underlying,
            toxicity_score=toxicity,
            max_position_size=self.config.max_position_per_market,
        )

        if signal is None:
            return

        self._signals_generated += 1

        # Callback
        if self._on_signal:
            self._on_signal(signal)

        # Check if tradeable
        if not signal.is_tradeable:
            return

        # Risk check
        if self.config.toxicity_enabled:
            assessment = self.risk_manager.evaluate_signal(signal, state.orderbook)
            if not assessment.approved:
                self._signals_filtered += 1
                return

        # Try to execute
        self._execute_signal(signal, state)

    def _execute_signal(self, signal: ArbitrageSignal, state: MarketState):
        """Execute a trading signal."""
        from src.data_feed.schemas import Order

        # Create order
        order = Order(
            market_ticker=signal.market_ticker,
            side=signal.side,
            price=int(signal.kalshi_mid) if signal.side == Side.BUY else int(signal.kalshi_mid),
            quantity=signal.recommended_size,
            client_order_id=f"bt_{signal.timestamp.timestamp()}",
        )

        # Simulate fill
        result = self.fill_model.try_fill(order, state.orderbook, signal.timestamp)

        if result.filled and result.fill:
            self._signals_traded += 1
            self._process_fill(result.fill, signal.market_ticker, signal)

    def _process_fill(
        self,
        fill,
        market: str,
        signal: Optional[ArbitrageSignal] = None,
    ):
        """Process a simulated fill."""
        from src.data_feed.schemas import Fill

        state = self._market_states.get(market)
        if not state:
            return

        # Update position
        old_pos = state.position
        if fill.side == Side.BUY:
            new_pos = old_pos + fill.quantity
        else:
            new_pos = old_pos - fill.quantity

        # Calculate P&L for closing portion
        pnl = 0.0
        if old_pos != 0 and ((old_pos > 0 and fill.side == Side.SELL) or (old_pos < 0 and fill.side == Side.BUY)):
            close_qty = min(abs(old_pos), fill.quantity)
            price_diff = fill.price - state.avg_entry
            if old_pos > 0:
                pnl = price_diff * close_qty / 100
            else:
                pnl = -price_diff * close_qty / 100

        # Update state
        if new_pos == 0:
            state.position = 0
            state.avg_entry = 0.0
        elif (old_pos >= 0 and new_pos > old_pos) or (old_pos <= 0 and new_pos < old_pos):
            # Adding to position
            if old_pos == 0:
                state.avg_entry = fill.price
            else:
                total_cost = state.avg_entry * abs(old_pos) + fill.price * fill.quantity
                state.avg_entry = total_cost / abs(new_pos)
            state.position = new_pos
        else:
            state.position = new_pos

        # Create backtest trade record
        trade = BacktestTrade(
            timestamp=fill.timestamp,
            market_ticker=market,
            side=fill.side,
            entry_price=fill.price,
            exit_price=None if new_pos != 0 else fill.price,
            quantity=fill.quantity,
            signal_edge=signal.net_edge if signal else 0.0,
            realized_edge=(pnl / fill.quantity * 100) if fill.quantity > 0 else 0.0,
            slippage=0.0,  # Would need expected price to calculate
            fees=fill.fee,
            pnl=pnl - fill.fee,
        )

        self._trades.append(trade)
        self.metrics.add_trade(trade)

        # Callback
        if self._on_trade:
            self._on_trade(trade)

    def _create_market_object(
        self,
        ticker: str,
        orderbook: KalshiOrderbook,
    ) -> Optional[KalshiMarket]:
        """Create market object from ticker."""
        # Parse ticker for strike price
        # Format: INXD-24JAN15-B5850 means "above 5850"
        parts = ticker.split("-")
        if len(parts) < 3:
            return None

        try:
            strike_str = parts[2]
            if strike_str.startswith("B"):
                strike = float(strike_str[1:])
            elif strike_str.startswith("T"):
                strike = float(strike_str[1:])
            else:
                return None

            # Parse expiry from ticker
            date_part = parts[1]
            # Simple approximation - in production get from API
            expiry = datetime.utcnow()

            return KalshiMarket(
                ticker=ticker,
                event_ticker=parts[0],
                title=f"S&P 500 {'above' if strike_str.startswith('B') else 'below'} {strike}",
                strike_price=strike,
                expiry=expiry,
                status=MarketStatus.OPEN,
            )
        except (ValueError, IndexError):
            return None

    def _close_all_positions(self):
        """Close all open positions at end of backtest."""
        for market, state in self._market_states.items():
            if state.position != 0 and state.orderbook:
                # Close at mid price
                mid = state.orderbook.mid_price or 50

                pnl = 0.0
                if state.position > 0:
                    pnl = (mid - state.avg_entry) * state.position / 100
                else:
                    pnl = (state.avg_entry - mid) * abs(state.position) / 100

                fee = abs(state.position) * self.config.taker_fee_per_contract

                trade = BacktestTrade(
                    timestamp=datetime.utcnow(),
                    market_ticker=market,
                    side=Side.SELL if state.position > 0 else Side.BUY,
                    entry_price=int(state.avg_entry),
                    exit_price=int(mid),
                    quantity=abs(state.position),
                    signal_edge=0.0,
                    realized_edge=0.0,
                    slippage=0.0,
                    fees=fee,
                    pnl=pnl - fee,
                )

                self._trades.append(trade)
                self.metrics.add_trade(trade)

    def reset(self):
        """Reset engine state for new backtest."""
        self._market_states.clear()
        self._underlying = None
        self._trades.clear()
        self._signals_generated = 0
        self._signals_traded = 0
        self._signals_filtered = 0
        self.metrics.reset()
        self.event_engine.clear()
        self.risk_manager.reset()
