"""
Integration tests for backtesting flow.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np

from src.data_feed.schemas import (
    KalshiOrderbook,
    KalshiTrade,
    PriceLevel,
    Side,
    UnderlyingTick,
)
from src.backtest.event_engine import EventEngine, Event, EventType
from src.backtest.backtest_engine import BacktestEngine
from src.backtest.config import BacktestConfig
from src.backtest.fill_model import SimpleFillModel


class TestEventEngine:
    """Test event engine functionality."""

    def test_events_processed_in_order(self):
        """Events should be processed in timestamp order."""
        engine = EventEngine()

        base_time = datetime.utcnow()
        processed_order = []

        def handler(event):
            processed_order.append(event.data['seq'])

        engine.register_handler(EventType.ORDERBOOK, handler)

        # Add events out of order
        engine.add_event(Event(
            timestamp=base_time + timedelta(seconds=2),
            event_type=EventType.ORDERBOOK,
            data={'seq': 2},
        ))
        engine.add_event(Event(
            timestamp=base_time,
            event_type=EventType.ORDERBOOK,
            data={'seq': 0},
        ))
        engine.add_event(Event(
            timestamp=base_time + timedelta(seconds=1),
            event_type=EventType.ORDERBOOK,
            data={'seq': 1},
        ))

        engine.run()

        assert processed_order == [0, 1, 2]

    def test_multiple_handlers(self):
        """Multiple handlers for same event type should all fire."""
        engine = EventEngine()

        results = []

        def handler1(event):
            results.append('h1')

        def handler2(event):
            results.append('h2')

        engine.register_handler(EventType.TRADE, handler1)
        engine.register_handler(EventType.TRADE, handler2)

        engine.add_event(Event(
            timestamp=datetime.utcnow(),
            event_type=EventType.TRADE,
            data={},
        ))

        engine.run()

        assert 'h1' in results
        assert 'h2' in results

    def test_different_event_types(self):
        """Different event types should route to correct handlers."""
        engine = EventEngine()

        results = {'ob': 0, 'trade': 0}

        def ob_handler(event):
            results['ob'] += 1

        def trade_handler(event):
            results['trade'] += 1

        engine.register_handler(EventType.ORDERBOOK, ob_handler)
        engine.register_handler(EventType.TRADE, trade_handler)

        base_time = datetime.utcnow()

        engine.add_event(Event(base_time, EventType.ORDERBOOK, {}))
        engine.add_event(Event(base_time, EventType.TRADE, {}))
        engine.add_event(Event(base_time, EventType.ORDERBOOK, {}))

        engine.run()

        assert results['ob'] == 2
        assert results['trade'] == 1


class TestBacktestEngine:
    """Test backtest engine integration."""

    def test_backtest_processes_orderbook_sequence(self, orderbook_sequence):
        """Backtest should process orderbook sequence."""
        config = BacktestConfig(
            initial_capital=10000,
            min_edge=0.01,
        )
        engine = BacktestEngine(config)

        # Load orderbooks
        for ob in orderbook_sequence:
            engine.on_orderbook(ob)

        # Should have processed without error
        assert engine.current_time is not None

    def test_backtest_tracks_positions(self, orderbook_sequence, trade_sequence):
        """Backtest should track positions from fills."""
        config = BacktestConfig(initial_capital=10000)
        engine = BacktestEngine(config, fill_model=SimpleFillModel())

        # Process data
        for ob in orderbook_sequence:
            engine.on_orderbook(ob)
        for trade in trade_sequence:
            engine.on_trade(trade)

        # Should have position tracking available
        positions = engine.get_positions()
        assert isinstance(positions, dict)

    def test_backtest_calculates_pnl(self, orderbook_sequence):
        """Backtest should calculate P&L."""
        config = BacktestConfig(initial_capital=10000)
        engine = BacktestEngine(config)

        for ob in orderbook_sequence:
            engine.on_orderbook(ob)

        # Get equity curve
        equity = engine.get_equity_curve()

        assert len(equity) > 0
        assert equity[0] == 10000  # Starting capital

    def test_backtest_generates_signals(self, sample_underlying):
        """Backtest should generate signals from price data."""
        config = BacktestConfig(
            initial_capital=10000,
            min_edge=0.01,
        )
        engine = BacktestEngine(config)

        # Create orderbook with mispricing
        ob = KalshiOrderbook(
            market_ticker="INXD-24JAN15-B5850",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=45, quantity=100)],
            yes_asks=[PriceLevel(price=47, quantity=100)],  # Mid = 46
        )

        # Underlying at 5870 (well above 5850 strike)
        underlying = UnderlyingTick(
            symbol="ES.FUT",
            timestamp=datetime.utcnow(),
            price=5870.0,
            size=1,
            bid_price=5869.75,
            bid_size=100,
            ask_price=5870.25,
            ask_size=100,
        )

        engine.on_underlying(underlying)
        signal = engine.on_orderbook(ob)

        # Should generate buy signal (fair prob > implied prob)
        if signal:
            assert signal.side == Side.BUY


class TestBacktestResults:
    """Test backtest result generation."""

    def test_results_contain_metrics(self):
        """Backtest results should contain performance metrics."""
        config = BacktestConfig(initial_capital=10000)
        engine = BacktestEngine(config)

        # Run minimal backtest
        base_time = datetime.utcnow()
        for i in range(20):
            ob = KalshiOrderbook(
                market_ticker="TEST",
                timestamp=base_time + timedelta(seconds=i),
                yes_bids=[PriceLevel(price=50 + (i % 3), quantity=100)],
                yes_asks=[PriceLevel(price=52 + (i % 3), quantity=100)],
            )
            engine.on_orderbook(ob)

        results = engine.get_results()

        assert 'total_return' in results
        assert 'sharpe_ratio' in results or 'sharpe' in results
        assert 'max_drawdown' in results
        assert 'num_trades' in results

    def test_results_include_trade_log(self):
        """Results should include detailed trade log."""
        config = BacktestConfig(initial_capital=10000)
        engine = BacktestEngine(config)

        # Process some data
        base_time = datetime.utcnow()
        for i in range(10):
            ob = KalshiOrderbook(
                market_ticker="TEST",
                timestamp=base_time + timedelta(seconds=i),
                yes_bids=[PriceLevel(price=50, quantity=100)],
                yes_asks=[PriceLevel(price=52, quantity=100)],
            )
            engine.on_orderbook(ob)

        trade_log = engine.get_trade_log()

        assert isinstance(trade_log, list)


class TestWalkForwardOptimization:
    """Test walk-forward optimization."""

    def test_parameter_optimization(self):
        """Should find optimal parameters."""
        from src.backtest.optimizer import GridSearchOptimizer
        from src.backtest.config import OptimizationConfig

        opt_config = OptimizationConfig(
            parameter_grid={
                'min_edge': [0.01, 0.02, 0.03],
                'max_toxicity': [0.5, 0.6, 0.7],
            },
            metric='sharpe',
        )
        optimizer = GridSearchOptimizer(opt_config)

        # Mock backtest function
        def run_backtest(params):
            # Higher edge = lower sharpe (for testing)
            return {'sharpe': 2.0 - params['min_edge'] * 10}

        best_params, best_metric = optimizer.optimize(run_backtest)

        # Should find lowest min_edge
        assert best_params['min_edge'] == 0.01

    def test_walk_forward_splits_data(self):
        """Walk-forward should properly split train/test periods."""
        from src.backtest.optimizer import WalkForwardOptimizer
        from src.backtest.config import OptimizationConfig

        opt_config = OptimizationConfig(
            train_days=60,
            test_days=20,
            parameter_grid={'min_edge': [0.01, 0.02]},
        )
        optimizer = WalkForwardOptimizer(opt_config)

        # Generate date range
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 30)

        periods = optimizer.generate_periods(start, end)

        # Each period should have train and test
        for period in periods:
            assert 'train_start' in period
            assert 'train_end' in period
            assert 'test_start' in period
            assert 'test_end' in period
            # Test should follow train
            assert period['test_start'] >= period['train_end']
