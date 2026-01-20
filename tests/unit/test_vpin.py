"""
Unit tests for VPIN calculator.
"""

import pytest
from datetime import datetime

from src.data_feed.schemas import KalshiTrade, Side
from src.risk.vpin import VPINCalculator


class TestVPINCalculator:
    """Tests for VPINCalculator."""

    def test_insufficient_data_returns_none(self, sample_trade):
        """Should return None with insufficient data."""
        vpin = VPINCalculator(bucket_size=100, num_buckets=50)
        vpin.update(sample_trade)

        assert vpin.get_vpin(sample_trade.market_ticker) is None

    def test_balanced_flow_low_vpin(self):
        """Balanced buy/sell flow should produce low VPIN."""
        vpin = VPINCalculator(bucket_size=10, num_buckets=5)
        market = "TEST"
        ts = datetime.utcnow()

        # Create balanced trades (50% buy, 50% sell)
        for i in range(100):
            trade = KalshiTrade(
                market_ticker=market,
                timestamp=ts,
                price=50,
                quantity=5,
                taker_side=Side.BUY if i % 2 == 0 else Side.SELL,
            )
            vpin.update(trade)

        result = vpin.get_vpin(market)

        # Perfectly balanced should give VPIN close to 0
        assert result is not None
        assert result < 0.3, "Balanced flow should have low VPIN"

    def test_one_sided_flow_high_vpin(self):
        """One-sided flow should produce high VPIN."""
        vpin = VPINCalculator(bucket_size=10, num_buckets=5)
        market = "TEST"
        ts = datetime.utcnow()

        # All buys
        for i in range(100):
            trade = KalshiTrade(
                market_ticker=market,
                timestamp=ts,
                price=50,
                quantity=5,
                taker_side=Side.BUY,
            )
            vpin.update(trade)

        result = vpin.get_vpin(market)

        # One-sided should give VPIN close to 1
        assert result is not None
        assert result > 0.7, "One-sided flow should have high VPIN"

    def test_state_tracking(self, trade_sequence):
        """State should track buckets and volumes correctly."""
        vpin = VPINCalculator(bucket_size=20, num_buckets=10)

        for trade in trade_sequence:
            vpin.update(trade)

        state = vpin.get_state("INXD-24JAN15-B5850")

        assert state is not None
        assert state.total_volume > 0
        assert state.num_buckets >= 1

    def test_toxicity_score_range(self, trade_sequence):
        """Toxicity score should be in [0, 1] range."""
        vpin = VPINCalculator(bucket_size=20, num_buckets=5)

        for trade in trade_sequence:
            vpin.update(trade)

        score = vpin.get_toxicity_score("INXD-24JAN15-B5850")

        assert 0 <= score <= 1

    def test_trade_direction_detection(self):
        """Should detect dominant trade direction."""
        vpin = VPINCalculator(bucket_size=10, num_buckets=5)
        market = "TEST"
        ts = datetime.utcnow()

        # Mostly buys
        for i in range(50):
            trade = KalshiTrade(
                market_ticker=market,
                timestamp=ts,
                price=50,
                quantity=5,
                taker_side=Side.BUY if i % 4 != 0 else Side.SELL,
            )
            vpin.update(trade)

        direction = vpin.get_trade_direction(market)

        assert direction == Side.BUY

    def test_reset_clears_state(self, sample_trade):
        """Reset should clear all state."""
        vpin = VPINCalculator(bucket_size=5, num_buckets=2)

        for _ in range(20):
            vpin.update(sample_trade)

        vpin.reset()

        assert vpin.get_vpin(sample_trade.market_ticker) is None
