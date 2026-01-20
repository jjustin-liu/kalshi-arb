"""
Unit tests for ExecutionEngine and related components.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.data_feed.schemas import (
    KalshiOrderbook,
    PriceLevel,
    Side,
    ArbitrageSignal,
    Fill,
)
from src.execution.config import ExecutionConfig, TWAPConfig
from src.execution.execution_engine import ExecutionEngine
from src.execution.order_manager import OrderManager
from src.execution.smart_orders import TWAPExecutor, IcebergExecutor


class TestExecutionEngine:
    """Tests for ExecutionEngine."""

    def test_execute_rejects_low_liquidity(self, sample_signal):
        """Should reject execution when liquidity is insufficient."""
        config = ExecutionConfig(min_depth=1000)  # Very high requirement
        engine = ExecutionEngine(config=config, kalshi_client=None)

        ob = KalshiOrderbook(
            market_ticker=sample_signal.market_ticker,
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=10)],
            yes_asks=[PriceLevel(price=52, quantity=10)],  # Low depth
        )

        result = engine.evaluate_execution(sample_signal, ob)

        assert not result.should_execute
        assert "liquidity" in result.rejection_reason.lower()

    def test_execute_rejects_high_toxicity(self, sample_signal):
        """Should reject execution when toxicity is too high."""
        config = ExecutionConfig(max_toxicity=0.1)  # Very low threshold
        engine = ExecutionEngine(config=config, kalshi_client=None)

        # Create signal with high toxicity
        toxic_signal = ArbitrageSignal(
            timestamp=datetime.utcnow(),
            market_ticker=sample_signal.market_ticker,
            underlying_symbol="ES.FUT",
            underlying_price=5865.0,
            strike_price=5850.0,
            kalshi_mid=56.0,
            fair_probability=0.62,
            implied_probability=0.56,
            basis=0.06,
            expected_fees=0.007,
            expected_slippage=0.01,
            net_edge=0.043,
            toxicity_score=0.8,  # High toxicity
            volatility=0.15,
            time_to_expiry=1 / 365,
            side=Side.BUY,
            recommended_size=20,
            confidence=0.75,
        )

        ob = KalshiOrderbook(
            market_ticker=toxic_signal.market_ticker,
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[PriceLevel(price=52, quantity=100)],
        )

        result = engine.evaluate_execution(toxic_signal, ob)

        assert not result.should_execute
        assert "toxic" in result.rejection_reason.lower()

    def test_selects_twap_for_large_orders(self, sample_signal):
        """Should select TWAP strategy for large orders."""
        config = ExecutionConfig(
            twap_threshold=10,
            twap_config=TWAPConfig(num_slices=5),
        )
        engine = ExecutionEngine(config=config, kalshi_client=None)

        # Large order signal
        large_signal = ArbitrageSignal(
            timestamp=datetime.utcnow(),
            market_ticker=sample_signal.market_ticker,
            underlying_symbol="ES.FUT",
            underlying_price=5865.0,
            strike_price=5850.0,
            kalshi_mid=56.0,
            fair_probability=0.62,
            implied_probability=0.56,
            basis=0.06,
            expected_fees=0.007,
            expected_slippage=0.01,
            net_edge=0.043,
            toxicity_score=0.2,
            volatility=0.15,
            time_to_expiry=1 / 365,
            side=Side.BUY,
            recommended_size=50,  # Large order
            confidence=0.75,
        )

        ob = KalshiOrderbook(
            market_ticker=large_signal.market_ticker,
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=200)],
            yes_asks=[PriceLevel(price=52, quantity=200)],
        )

        result = engine.evaluate_execution(large_signal, ob)

        assert result.should_execute
        assert result.strategy == "twap"

    def test_position_tracking(self, sample_signal, sample_orderbook):
        """Should track positions after execution."""
        engine = ExecutionEngine(config=ExecutionConfig(), kalshi_client=None)

        # Simulate a fill
        fill = Fill(
            order_id="test-123",
            market_ticker=sample_signal.market_ticker,
            side=Side.BUY,
            price=52.0,
            quantity=10,
            timestamp=datetime.utcnow(),
            fees=0.07,
        )

        engine.record_fill(fill)

        position = engine.get_position(sample_signal.market_ticker)

        assert position is not None
        assert position.quantity == 10
        assert position.side == Side.BUY


class TestTWAPExecutor:
    """Tests for TWAP execution."""

    def test_slice_calculation(self):
        """Should calculate correct slice sizes."""
        config = TWAPConfig(num_slices=5, interval_seconds=60)
        executor = TWAPExecutor(config)

        slices = executor.calculate_slices(total_quantity=100)

        assert len(slices) == 5
        assert sum(slices) == 100
        # Each slice should be roughly 20
        assert all(15 <= s <= 25 for s in slices)

    def test_randomizes_slice_timing(self):
        """Should add randomness to timing."""
        config = TWAPConfig(
            num_slices=5,
            interval_seconds=60,
            randomize_timing=True,
            timing_variance=0.2,
        )
        executor = TWAPExecutor(config)

        timings = [executor.get_next_interval() for _ in range(10)]

        # Should have variance
        assert len(set(timings)) > 1
        # Should be within bounds
        assert all(48 <= t <= 72 for t in timings)


class TestIcebergExecutor:
    """Tests for Iceberg execution."""

    def test_visible_quantity(self):
        """Should only show portion of order."""
        from src.execution.config import IcebergConfig

        config = IcebergConfig(visible_pct=0.2)
        executor = IcebergExecutor(config)

        visible = executor.get_visible_quantity(total=100)

        assert visible == 20

    def test_refills_on_fill(self):
        """Should refill visible portion after fill."""
        from src.execution.config import IcebergConfig

        config = IcebergConfig(visible_pct=0.2, refill_threshold=0.5)
        executor = IcebergExecutor(config)

        # Start with 100, show 20
        executor.initialize(total_quantity=100)
        assert executor.visible_quantity == 20

        # Simulate partial fill of 15
        executor.on_partial_fill(filled_quantity=15)

        # Should refill since remaining visible (5) < threshold (10)
        assert executor.visible_quantity == 20
        assert executor.remaining_quantity == 85


class TestOrderManager:
    """Tests for OrderManager."""

    def test_tracks_pending_orders(self):
        """Should track pending orders."""
        manager = OrderManager(kalshi_client=None)

        order_id = manager.create_order(
            market_ticker="TEST",
            side=Side.BUY,
            price=50,
            quantity=10,
        )

        assert manager.has_pending_order(order_id)
        assert manager.get_order_status(order_id) == "pending"

    def test_handles_fill_callback(self):
        """Should invoke fill callback on fill."""
        manager = OrderManager(kalshi_client=None)
        fills_received = []

        def on_fill(fill):
            fills_received.append(fill)

        manager.on_fill = on_fill

        order_id = manager.create_order(
            market_ticker="TEST",
            side=Side.BUY,
            price=50,
            quantity=10,
        )

        # Simulate fill
        fill = Fill(
            order_id=order_id,
            market_ticker="TEST",
            side=Side.BUY,
            price=50,
            quantity=10,
            timestamp=datetime.utcnow(),
            fees=0.07,
        )
        manager.process_fill(fill)

        assert len(fills_received) == 1
        assert fills_received[0].order_id == order_id

    def test_cancels_stale_orders(self):
        """Should cancel orders that exceed timeout."""
        from src.execution.config import ExecutionConfig

        config = ExecutionConfig(partial_fill_timeout=0.001)  # Very short
        manager = OrderManager(kalshi_client=None, config=config)

        order_id = manager.create_order(
            market_ticker="TEST",
            side=Side.BUY,
            price=50,
            quantity=10,
        )

        import time
        time.sleep(0.01)

        stale = manager.get_stale_orders()

        assert order_id in stale
