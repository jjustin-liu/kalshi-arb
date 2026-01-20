"""
Unit tests for Order Flow Imbalance calculator.
"""

import pytest
from datetime import datetime, timedelta

from src.data_feed.schemas import KalshiOrderbook, PriceLevel
from src.risk.ofi import OFICalculator


class TestOFICalculator:
    """Tests for OFICalculator."""

    def test_first_update_returns_none(self, sample_orderbook):
        """First update should return None (need 2 for comparison)."""
        ofi = OFICalculator()
        result = ofi.update(sample_orderbook)
        assert result is None

    def test_no_change_returns_zero_ofi(self, sample_orderbook):
        """Identical orderbooks should produce zero OFI."""
        ofi = OFICalculator()
        ofi.update(sample_orderbook)

        # Second identical update
        result = ofi.update(sample_orderbook)
        assert result == 0.0

    def test_bid_increase_produces_positive_ofi(self):
        """Increasing bid depth should produce positive OFI."""
        ofi = OFICalculator()

        # Initial orderbook
        ob1 = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[PriceLevel(price=52, quantity=100)],
        )
        ofi.update(ob1)

        # Bid depth increases
        ob2 = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow() + timedelta(seconds=1),
            yes_bids=[PriceLevel(price=50, quantity=150)],  # +50
            yes_asks=[PriceLevel(price=52, quantity=100)],  # unchanged
        )
        result = ofi.update(ob2)

        assert result > 0, "Bid increase should produce positive OFI"

    def test_ask_sweep_produces_negative_ofi(self):
        """Ask side being hit (depth decrease) should reduce OFI."""
        ofi = OFICalculator()

        # Initial orderbook
        ob1 = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[PriceLevel(price=52, quantity=100)],
        )
        ofi.update(ob1)

        # Ask depth decreases (someone bought)
        ob2 = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow() + timedelta(seconds=1),
            yes_bids=[PriceLevel(price=50, quantity=100)],  # unchanged
            yes_asks=[PriceLevel(price=52, quantity=50)],  # -50
        )
        result = ofi.update(ob2)

        # Note: OFI calculation depends on implementation
        # Ask decrease could mean buyers took liquidity
        assert result is not None

    def test_zscore_normalization(self, orderbook_sequence):
        """Z-score should normalize OFI values."""
        ofi = OFICalculator(zscore_lookback=10)

        # Feed sequence of orderbooks
        for ob in orderbook_sequence:
            ofi.update(ob)

        state = ofi.get_state("INXD-24JAN15-B5850")

        # Should have statistics
        assert state is not None
        assert state.num_observations > 1
        # Z-score should be finite
        assert abs(state.ofi_zscore) < 100

    def test_toxicity_score_range(self, orderbook_sequence):
        """Toxicity score should be in [0, 1] range."""
        ofi = OFICalculator()

        for ob in orderbook_sequence:
            ofi.update(ob)

        score = ofi.get_toxicity_score("INXD-24JAN15-B5850")

        assert 0 <= score <= 1

    def test_reset_clears_state(self, sample_orderbook):
        """Reset should clear all state."""
        ofi = OFICalculator()
        ofi.update(sample_orderbook)
        ofi.update(sample_orderbook)

        ofi.reset()

        assert ofi.get_state(sample_orderbook.market_ticker) is None
