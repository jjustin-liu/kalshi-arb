"""
Unit tests for LiquidityAnalyzer.
"""

import pytest
from datetime import datetime

from src.data_feed.schemas import KalshiOrderbook, PriceLevel, Side
from src.execution.liquidity import LiquidityAnalyzer


class TestLiquidityAnalyzer:
    """Tests for LiquidityAnalyzer."""

    def test_can_execute_with_good_liquidity(self, sample_orderbook):
        """Should allow execution with sufficient liquidity."""
        analyzer = LiquidityAnalyzer(
            max_spread_cents=5,
            min_depth=10,
            max_impact_pct=0.05,
        )

        check = analyzer.can_execute(sample_orderbook, Side.BUY, quantity=10)

        assert check.can_execute
        assert check.reason == "OK"

    def test_rejects_wide_spread(self):
        """Should reject when spread is too wide."""
        analyzer = LiquidityAnalyzer(max_spread_cents=2)

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=45, quantity=100)],
            yes_asks=[PriceLevel(price=55, quantity=100)],  # 10 cent spread
        )

        check = analyzer.can_execute(ob, Side.BUY, quantity=10)

        assert not check.can_execute
        assert "Spread" in check.reason

    def test_rejects_insufficient_depth(self):
        """Should reject when depth is too low."""
        analyzer = LiquidityAnalyzer(min_depth=100)

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=10)],
            yes_asks=[PriceLevel(price=52, quantity=10)],  # Only 10 contracts
        )

        check = analyzer.can_execute(ob, Side.BUY, quantity=10)

        assert not check.can_execute
        assert "depth" in check.reason.lower()

    def test_rejects_empty_orderbook(self):
        """Should reject empty orderbook."""
        analyzer = LiquidityAnalyzer()

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[],
            yes_asks=[],
        )

        check = analyzer.can_execute(ob, Side.BUY, quantity=10)

        assert not check.can_execute
        assert "Empty" in check.reason

    def test_calculates_spread_correctly(self, sample_orderbook):
        """Should calculate spread correctly."""
        analyzer = LiquidityAnalyzer()

        check = analyzer.can_execute(sample_orderbook, Side.BUY, quantity=10)

        assert check.spread == sample_orderbook.spread
        assert check.best_bid == sample_orderbook.best_bid
        assert check.best_ask == sample_orderbook.best_ask

    def test_calculates_depth_correctly(self, sample_orderbook):
        """Should calculate total depth correctly."""
        analyzer = LiquidityAnalyzer()

        check = analyzer.can_execute(sample_orderbook, Side.BUY, quantity=10)

        expected_bid_depth = sum(lvl.quantity for lvl in sample_orderbook.yes_bids)
        expected_ask_depth = sum(lvl.quantity for lvl in sample_orderbook.yes_asks)

        assert check.bid_depth == expected_bid_depth
        assert check.ask_depth == expected_ask_depth

    def test_estimates_market_impact(self):
        """Should estimate market impact."""
        analyzer = LiquidityAnalyzer()

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[
                PriceLevel(price=52, quantity=10),
                PriceLevel(price=53, quantity=20),
                PriceLevel(price=54, quantity=30),
            ],
        )

        # Large order should have higher impact
        check_small = analyzer.can_execute(ob, Side.BUY, quantity=5)
        check_large = analyzer.can_execute(ob, Side.BUY, quantity=30)

        assert check_large.estimated_impact >= check_small.estimated_impact

    def test_recommended_size_respects_depth(self, sample_orderbook):
        """Recommended size should not exceed available depth."""
        analyzer = LiquidityAnalyzer(max_pct_of_depth=0.5)

        recommended = analyzer.get_recommended_size(
            sample_orderbook,
            Side.BUY,
            target_size=1000,
        )

        # Should be limited by depth
        ask_depth = sample_orderbook.yes_asks[0].quantity if sample_orderbook.yes_asks else 0
        assert recommended <= ask_depth

    def test_effective_price_calculation(self):
        """Should calculate effective price correctly."""
        analyzer = LiquidityAnalyzer()

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[
                PriceLevel(price=52, quantity=10),
                PriceLevel(price=53, quantity=20),
            ],
        )

        # Small order should get best price
        effective_small = analyzer.get_effective_price(ob, Side.BUY, quantity=5)
        assert effective_small == 52.0

        # Large order should have worse effective price
        effective_large = analyzer.get_effective_price(ob, Side.BUY, quantity=25)
        assert effective_large > 52.0  # Walk through multiple levels
