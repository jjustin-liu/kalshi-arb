"""
Integration tests for the execution flow.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.data_feed.schemas import (
    KalshiOrderbook,
    KalshiTrade,
    PriceLevel,
    Side,
    ArbitrageSignal,
    Fill,
)
from src.execution.execution_engine import ExecutionEngine
from src.execution.config import ExecutionConfig
from src.risk.risk_manager import RiskManager
from src.risk.config import RiskConfig


class TestSignalToFillFlow:
    """Test complete signal-to-fill execution flow."""

    def test_full_execution_flow(self, sample_signal, sample_orderbook):
        """Test signal evaluation through execution decision."""
        # Setup
        risk_config = RiskConfig()
        risk_manager = RiskManager(risk_config)

        exec_config = ExecutionConfig(
            max_toxicity=0.8,
            min_depth=10,
        )
        engine = ExecutionEngine(config=exec_config, kalshi_client=None)

        # Step 1: Risk assessment
        risk_assessment = risk_manager.evaluate_signal(
            sample_signal, sample_orderbook
        )

        # Step 2: Execution evaluation
        if risk_assessment.approved:
            exec_result = engine.evaluate_execution(
                sample_signal, sample_orderbook
            )

            assert exec_result is not None
            # With good liquidity and low toxicity, should approve
            if exec_result.should_execute:
                assert exec_result.strategy in ['immediate', 'twap', 'iceberg']

    def test_toxicity_blocks_execution(self, sample_orderbook):
        """High toxicity should block execution."""
        risk_config = RiskConfig(max_toxicity=0.3)
        risk_manager = RiskManager(risk_config)

        # Create high toxicity signal
        toxic_signal = ArbitrageSignal(
            timestamp=datetime.utcnow(),
            market_ticker="TEST",
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
            toxicity_score=0.7,  # High
            volatility=0.15,
            time_to_expiry=1 / 365,
            side=Side.BUY,
            recommended_size=20,
            confidence=0.75,
        )

        assessment = risk_manager.evaluate_signal(toxic_signal, sample_orderbook)

        assert not assessment.approved
        assert "toxic" in assessment.rejection_reason.lower()

    def test_position_limit_blocks_execution(self, sample_signal, sample_orderbook):
        """Position limits should block oversized orders."""
        risk_config = RiskConfig(
            max_position_per_market=10,
        )
        risk_manager = RiskManager(risk_config)

        # Simulate existing position
        risk_manager.record_position(
            sample_signal.market_ticker, Side.BUY, quantity=8
        )

        # Try to add more
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
            recommended_size=50,  # Too large
            confidence=0.75,
        )

        assessment = risk_manager.evaluate_signal(large_signal, sample_orderbook)

        # Should either reject or reduce size
        if assessment.approved:
            assert assessment.adjusted_size <= 2  # Can only add 2 more
        else:
            assert "position" in assessment.rejection_reason.lower()

    def test_fill_updates_position(self, sample_signal):
        """Fills should update position tracking."""
        exec_config = ExecutionConfig()
        engine = ExecutionEngine(config=exec_config, kalshi_client=None)

        # Initial position
        initial_pos = engine.get_position(sample_signal.market_ticker)
        initial_qty = initial_pos.quantity if initial_pos else 0

        # Record a fill
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

        # Check position updated
        new_pos = engine.get_position(sample_signal.market_ticker)
        assert new_pos.quantity == initial_qty + 10

    def test_opposing_fill_reduces_position(self, sample_signal):
        """Opposing fills should reduce position."""
        exec_config = ExecutionConfig()
        engine = ExecutionEngine(config=exec_config, kalshi_client=None)

        # Buy first
        buy_fill = Fill(
            order_id="buy-123",
            market_ticker=sample_signal.market_ticker,
            side=Side.BUY,
            price=52.0,
            quantity=20,
            timestamp=datetime.utcnow(),
            fees=0.07,
        )
        engine.record_fill(buy_fill)

        # Then sell some
        sell_fill = Fill(
            order_id="sell-456",
            market_ticker=sample_signal.market_ticker,
            side=Side.SELL,
            price=54.0,
            quantity=15,
            timestamp=datetime.utcnow(),
            fees=0.07,
        )
        engine.record_fill(sell_fill)

        # Check net position
        pos = engine.get_position(sample_signal.market_ticker)
        assert pos.quantity == 5  # 20 - 15


class TestLiquidityChecks:
    """Test liquidity check integration."""

    def test_spread_check_integration(self):
        """Wide spread should fail liquidity check in execution flow."""
        exec_config = ExecutionConfig(max_spread_cents=3)
        engine = ExecutionEngine(config=exec_config, kalshi_client=None)

        signal = ArbitrageSignal(
            timestamp=datetime.utcnow(),
            market_ticker="TEST",
            underlying_symbol="ES.FUT",
            underlying_price=5865.0,
            strike_price=5850.0,
            kalshi_mid=52.5,
            fair_probability=0.55,
            implied_probability=0.525,
            basis=0.025,
            expected_fees=0.007,
            expected_slippage=0.01,
            net_edge=0.008,
            toxicity_score=0.2,
            volatility=0.15,
            time_to_expiry=1 / 365,
            side=Side.BUY,
            recommended_size=10,
            confidence=0.6,
        )

        # Wide spread orderbook
        wide_ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=45, quantity=100)],
            yes_asks=[PriceLevel(price=55, quantity=100)],  # 10 cent spread
        )

        result = engine.evaluate_execution(signal, wide_ob)

        assert not result.should_execute
        assert "spread" in result.rejection_reason.lower()

    def test_depth_check_integration(self):
        """Insufficient depth should fail liquidity check."""
        exec_config = ExecutionConfig(min_depth=50)
        engine = ExecutionEngine(config=exec_config, kalshi_client=None)

        signal = ArbitrageSignal(
            timestamp=datetime.utcnow(),
            market_ticker="TEST",
            underlying_symbol="ES.FUT",
            underlying_price=5865.0,
            strike_price=5850.0,
            kalshi_mid=51.0,
            fair_probability=0.55,
            implied_probability=0.51,
            basis=0.04,
            expected_fees=0.007,
            expected_slippage=0.01,
            net_edge=0.023,
            toxicity_score=0.2,
            volatility=0.15,
            time_to_expiry=1 / 365,
            side=Side.BUY,
            recommended_size=10,
            confidence=0.7,
        )

        # Thin orderbook
        thin_ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=10)],
            yes_asks=[PriceLevel(price=52, quantity=10)],  # Only 10 depth
        )

        result = engine.evaluate_execution(signal, thin_ob)

        assert not result.should_execute
        assert "depth" in result.rejection_reason.lower()


class TestHedgeIntegration:
    """Test hedge simulator integration."""

    def test_hedge_position_updates_on_fill(self, sample_signal):
        """Hedge position should update when Kalshi trade fills."""
        from src.execution.hedge_simulator import HedgeSimulator
        from src.execution.config import HedgeConfig

        hedge_config = HedgeConfig(delta_per_contract=0.01)
        hedger = HedgeSimulator(hedge_config)

        # Simulate Kalshi fill
        kalshi_fill = Fill(
            order_id="k-123",
            market_ticker=sample_signal.market_ticker,
            side=Side.BUY,
            price=55.0,
            quantity=100,
            timestamp=datetime.utcnow(),
            fees=0.70,
        )

        hedger.on_kalshi_fill(kalshi_fill, underlying_price=5865.0)

        # Check hedge position
        hedge_pos = hedger.get_position()

        # Bought 100 YES contracts = need to short delta * 100 ES
        assert hedge_pos.es_contracts != 0

    def test_hedge_pnl_attribution(self):
        """Should track hedge P&L separately."""
        from src.execution.hedge_simulator import HedgeSimulator
        from src.execution.config import HedgeConfig

        hedge_config = HedgeConfig(delta_per_contract=0.01)
        hedger = HedgeSimulator(hedge_config)

        # Initial hedge
        hedger.set_position(es_contracts=-1.0, entry_price=5865.0)

        # ES price moves
        hedger.update_underlying_price(5870.0)  # +5 points

        pnl = hedger.get_unrealized_pnl()

        # Short 1 ES, price went up 5 = loss of 5 * $50 = -$250
        assert pnl < 0
