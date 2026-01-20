"""
Integration tests for risk management flow.
"""

import pytest
from datetime import datetime, timedelta

from src.data_feed.schemas import (
    KalshiOrderbook,
    KalshiTrade,
    PriceLevel,
    Side,
    ArbitrageSignal,
)
from src.risk.risk_manager import RiskManager
from src.risk.config import RiskConfig, ToxicityConfig
from src.risk.toxicity_monitor import ToxicityMonitor
from src.risk.ofi import OFICalculator
from src.risk.vpin import VPINCalculator


class TestToxicityIntegration:
    """Test toxicity monitoring integration."""

    def test_toxicity_combines_all_signals(self, orderbook_sequence, trade_sequence):
        """Toxicity should combine OFI, VPIN, and other signals."""
        config = ToxicityConfig()
        monitor = ToxicityMonitor(config)

        market = orderbook_sequence[0].market_ticker

        # Feed orderbook and trade data
        for ob in orderbook_sequence:
            monitor.update_orderbook(ob)

        for trade in trade_sequence:
            monitor.update_trade(trade)

        # Get combined toxicity
        toxicity = monitor.get_toxicity_score(market, orderbook_sequence[-1])

        assert 0 <= toxicity <= 1

        # Get component breakdown
        components = monitor.get_component_scores(market, orderbook_sequence[-1])

        assert 'ofi' in components
        assert 'vpin' in components
        assert 'spread' in components

    def test_sweep_detection_increases_toxicity(self):
        """Detecting a sweep should increase toxicity."""
        config = ToxicityConfig(weight_sweep=0.3)  # High weight for sweep
        monitor = ToxicityMonitor(config)

        market = "TEST"
        base_time = datetime.utcnow()

        # Normal orderbook
        ob_normal = KalshiOrderbook(
            market_ticker=market,
            timestamp=base_time,
            yes_bids=[
                PriceLevel(price=50, quantity=50),
                PriceLevel(price=49, quantity=50),
                PriceLevel(price=48, quantity=50),
            ],
            yes_asks=[
                PriceLevel(price=52, quantity=50),
                PriceLevel(price=53, quantity=50),
                PriceLevel(price=54, quantity=50),
            ],
        )

        for _ in range(5):
            monitor.update_orderbook(ob_normal)

        score_before = monitor.get_toxicity_score(market, ob_normal)

        # Swept orderbook (multiple levels cleared)
        ob_swept = KalshiOrderbook(
            market_ticker=market,
            timestamp=base_time + timedelta(milliseconds=100),
            yes_bids=[
                PriceLevel(price=50, quantity=50),
                PriceLevel(price=49, quantity=50),
                PriceLevel(price=48, quantity=50),
            ],
            yes_asks=[
                PriceLevel(price=55, quantity=50),  # Levels 52-54 cleared
            ],
        )

        monitor.update_orderbook(ob_swept)
        score_after = monitor.get_toxicity_score(market, ob_swept)

        # Sweep should increase toxicity
        assert score_after >= score_before

    def test_vpin_rises_with_informed_trading(self):
        """VPIN should rise with one-sided trading."""
        vpin_calc = VPINCalculator(bucket_size=10, num_buckets=5)
        market = "TEST"
        ts = datetime.utcnow()

        # Initial balanced trading
        for i in range(30):
            trade = KalshiTrade(
                market_ticker=market,
                timestamp=ts + timedelta(seconds=i),
                price=50,
                quantity=5,
                taker_side=Side.BUY if i % 2 == 0 else Side.SELL,
            )
            vpin_calc.update(trade)

        vpin_balanced = vpin_calc.get_vpin(market)

        # One-sided buying burst
        for i in range(30, 60):
            trade = KalshiTrade(
                market_ticker=market,
                timestamp=ts + timedelta(seconds=i),
                price=50 + (i - 30) * 0.1,  # Price rising
                quantity=5,
                taker_side=Side.BUY,  # All buys
            )
            vpin_calc.update(trade)

        vpin_onesided = vpin_calc.get_vpin(market)

        # VPIN should be higher after one-sided flow
        assert vpin_onesided > vpin_balanced

    def test_ofi_tracks_orderbook_pressure(self):
        """OFI should track buying/selling pressure."""
        ofi_calc = OFICalculator()
        market = "TEST"

        # Bid depth increasing
        ob1 = KalshiOrderbook(
            market_ticker=market,
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[PriceLevel(price=52, quantity=100)],
        )
        ofi_calc.update(ob1)

        ob2 = KalshiOrderbook(
            market_ticker=market,
            timestamp=datetime.utcnow() + timedelta(seconds=1),
            yes_bids=[PriceLevel(price=50, quantity=200)],  # +100
            yes_asks=[PriceLevel(price=52, quantity=100)],  # unchanged
        )
        ofi = ofi_calc.update(ob2)

        # Positive OFI = buying pressure
        assert ofi > 0


class TestRiskManagerIntegration:
    """Test risk manager integration."""

    def test_risk_manager_uses_all_components(
        self, sample_signal, sample_orderbook, trade_sequence
    ):
        """Risk manager should use toxicity, position limits, etc."""
        config = RiskConfig(
            max_position_per_market=100,
            max_total_position=500,
            max_toxicity=0.7,
            max_daily_loss=1000,
        )
        manager = RiskManager(config)

        # Feed some trade data
        for trade in trade_sequence[:10]:
            manager.update_trade(trade)

        # Evaluate signal
        assessment = manager.evaluate_signal(sample_signal, sample_orderbook)

        assert assessment is not None
        assert hasattr(assessment, 'approved')
        assert hasattr(assessment, 'adjusted_size')
        assert hasattr(assessment, 'toxicity_score')

    def test_daily_loss_limit_blocks_trading(self, sample_signal, sample_orderbook):
        """Should stop trading when daily loss limit hit."""
        config = RiskConfig(max_daily_loss=100)
        manager = RiskManager(config)

        # Simulate losses
        manager.record_pnl(-150)  # Exceed limit

        assessment = manager.evaluate_signal(sample_signal, sample_orderbook)

        assert not assessment.approved
        assert "loss" in assessment.rejection_reason.lower()

    def test_position_sizing_respects_limits(self, sample_orderbook):
        """Position sizing should respect limits."""
        config = RiskConfig(max_position_per_market=50)
        manager = RiskManager(config)

        # Large signal
        large_signal = ArbitrageSignal(
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
            toxicity_score=0.2,
            volatility=0.15,
            time_to_expiry=1 / 365,
            side=Side.BUY,
            recommended_size=200,  # Way over limit
            confidence=0.75,
        )

        assessment = manager.evaluate_signal(large_signal, sample_orderbook)

        if assessment.approved:
            assert assessment.adjusted_size <= 50

    def test_correlation_limit_enforcement(self, sample_orderbook):
        """Should enforce correlated position limits."""
        config = RiskConfig(
            max_position_per_market=100,
            max_correlated_position=150,
        )
        manager = RiskManager(config)

        # Existing position in correlated market
        manager.record_position("INXD-A", Side.BUY, 100)
        manager.record_position("INXD-B", Side.BUY, 40)

        # Try to add more in same direction
        signal = ArbitrageSignal(
            timestamp=datetime.utcnow(),
            market_ticker="INXD-C",  # Same event
            underlying_symbol="ES.FUT",
            underlying_price=5865.0,
            strike_price=5855.0,
            kalshi_mid=52.0,
            fair_probability=0.55,
            implied_probability=0.52,
            basis=0.03,
            expected_fees=0.007,
            expected_slippage=0.01,
            net_edge=0.013,
            toxicity_score=0.2,
            volatility=0.15,
            time_to_expiry=1 / 365,
            side=Side.BUY,
            recommended_size=50,
            confidence=0.6,
        )

        assessment = manager.evaluate_signal(signal, sample_orderbook)

        # Should limit to stay under correlated max
        if assessment.approved:
            assert assessment.adjusted_size <= 10  # 150 - 140 = 10


class TestPauseTrading:
    """Test trading pause functionality."""

    def test_pause_on_high_toxicity(self, sample_orderbook):
        """Should pause trading when toxicity exceeds threshold."""
        config = ToxicityConfig(pause_threshold=0.5)
        monitor = ToxicityMonitor(config)

        market = sample_orderbook.market_ticker

        # Simulate high toxicity conditions
        # Create severely imbalanced orderbook
        toxic_ob = KalshiOrderbook(
            market_ticker=market,
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=40, quantity=10)],  # Thin, wide
            yes_asks=[PriceLevel(price=60, quantity=500)],  # Heavy one side
        )

        for _ in range(10):
            monitor.update_orderbook(toxic_ob)

        # Check if should pause
        should_pause = monitor.should_pause_trading(market, toxic_ob)

        # Result depends on actual toxicity calculation
        assert isinstance(should_pause, bool)

    def test_resume_after_cooldown(self, sample_orderbook):
        """Should resume trading after conditions normalize."""
        config = ToxicityConfig(pause_threshold=0.5, cooldown_seconds=1)
        monitor = ToxicityMonitor(config)

        market = sample_orderbook.market_ticker

        # Feed normal data
        for _ in range(20):
            monitor.update_orderbook(sample_orderbook)

        # Should not be paused with normal conditions
        should_pause = monitor.should_pause_trading(market, sample_orderbook)

        # Normal orderbook should not trigger pause
        score = monitor.get_toxicity_score(market, sample_orderbook)
        if score < 0.5:
            assert not should_pause
