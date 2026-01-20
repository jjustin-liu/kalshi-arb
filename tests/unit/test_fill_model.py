"""
Unit tests for fill simulation models.
"""

import pytest
from datetime import datetime

from src.data_feed.schemas import KalshiOrderbook, PriceLevel, Side
from src.backtest.fill_model import SimpleFillModel, QueueFillModel, ImpactFillModel


class TestSimpleFillModel:
    """Tests for SimpleFillModel."""

    def test_fills_at_best_price(self):
        """Should fill immediately at best available price."""
        model = SimpleFillModel()

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[PriceLevel(price=52, quantity=100)],
        )

        fill = model.simulate_fill(ob, Side.BUY, quantity=10, limit_price=55)

        assert fill is not None
        assert fill.price == 52  # Best ask
        assert fill.quantity == 10

    def test_respects_limit_price(self):
        """Should not fill if limit price is worse than market."""
        model = SimpleFillModel()

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[PriceLevel(price=52, quantity=100)],
        )

        # Try to buy at 51 when best ask is 52
        fill = model.simulate_fill(ob, Side.BUY, quantity=10, limit_price=51)

        assert fill is None

    def test_partial_fill_when_insufficient_depth(self):
        """Should partially fill when depth is insufficient."""
        model = SimpleFillModel(allow_partial=True)

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[PriceLevel(price=52, quantity=5)],  # Only 5 available
        )

        fill = model.simulate_fill(ob, Side.BUY, quantity=10, limit_price=55)

        assert fill is not None
        assert fill.quantity == 5  # Partial fill

    def test_sell_side_fills(self):
        """Should fill sells at best bid."""
        model = SimpleFillModel()

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[PriceLevel(price=52, quantity=100)],
        )

        fill = model.simulate_fill(ob, Side.SELL, quantity=10, limit_price=48)

        assert fill is not None
        assert fill.price == 50  # Best bid
        assert fill.quantity == 10


class TestQueueFillModel:
    """Tests for QueueFillModel with queue position simulation."""

    def test_queue_position_affects_fill_probability(self):
        """Orders further back in queue less likely to fill."""
        model = QueueFillModel(base_fill_rate=0.5)

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[PriceLevel(price=52, quantity=100)],
        )

        # Simulate many attempts
        fills_at_front = 0
        fills_at_back = 0

        for _ in range(100):
            # Front of queue (queue_position=0)
            fill = model.simulate_fill(
                ob, Side.BUY, quantity=10, limit_price=52, queue_position=0
            )
            if fill:
                fills_at_front += 1

            # Back of queue (queue_position=90)
            fill = model.simulate_fill(
                ob, Side.BUY, quantity=10, limit_price=52, queue_position=90
            )
            if fill:
                fills_at_back += 1

        # Front of queue should fill more often
        assert fills_at_front > fills_at_back

    def test_aggressive_orders_always_fill(self):
        """Crossing the spread should always fill."""
        model = QueueFillModel()

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[PriceLevel(price=52, quantity=100)],
        )

        # Buy at 53 (above best ask of 52)
        fill = model.simulate_fill(ob, Side.BUY, quantity=10, limit_price=53)

        assert fill is not None
        assert fill.price == 52  # Gets best ask price


class TestImpactFillModel:
    """Tests for ImpactFillModel with market impact."""

    def test_small_orders_minimal_impact(self):
        """Small orders should have minimal price impact."""
        model = ImpactFillModel(impact_coefficient=0.1)

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[
                PriceLevel(price=52, quantity=50),
                PriceLevel(price=53, quantity=50),
                PriceLevel(price=54, quantity=50),
            ],
        )

        fill = model.simulate_fill(ob, Side.BUY, quantity=5, limit_price=60)

        assert fill is not None
        # Small order should mostly fill at best price
        assert fill.price <= 52.5

    def test_large_orders_walk_book(self):
        """Large orders should walk through multiple price levels."""
        model = ImpactFillModel(impact_coefficient=0.1)

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[
                PriceLevel(price=52, quantity=10),
                PriceLevel(price=53, quantity=10),
                PriceLevel(price=54, quantity=10),
            ],
        )

        fill = model.simulate_fill(ob, Side.BUY, quantity=25, limit_price=60)

        assert fill is not None
        # Should have walked through multiple levels
        # Effective price should be weighted average
        assert fill.price > 52

    def test_calculates_slippage(self):
        """Should calculate slippage from mid price."""
        model = ImpactFillModel()

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[
                PriceLevel(price=52, quantity=10),
                PriceLevel(price=54, quantity=20),
            ],
        )

        fill = model.simulate_fill(ob, Side.BUY, quantity=20, limit_price=60)

        assert fill is not None
        # Mid is 51, fill price should be above that
        mid_price = 51
        slippage = fill.price - mid_price
        assert slippage > 0

    def test_impact_model_respects_available_liquidity(self):
        """Should not fill more than available liquidity."""
        model = ImpactFillModel()

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[
                PriceLevel(price=52, quantity=10),
                PriceLevel(price=53, quantity=10),
            ],
        )

        # Try to buy 50 when only 20 available
        fill = model.simulate_fill(ob, Side.BUY, quantity=50, limit_price=60)

        assert fill is not None
        assert fill.quantity <= 20


class TestFillModelIntegration:
    """Integration tests across fill models."""

    def test_all_models_return_consistent_structure(self, sample_orderbook):
        """All models should return fills with same structure."""
        models = [
            SimpleFillModel(),
            QueueFillModel(),
            ImpactFillModel(),
        ]

        for model in models:
            fill = model.simulate_fill(
                sample_orderbook,
                Side.BUY,
                quantity=5,
                limit_price=60,
            )

            assert fill is not None
            assert hasattr(fill, 'price')
            assert hasattr(fill, 'quantity')
            assert hasattr(fill, 'timestamp')
            assert hasattr(fill, 'side')

    def test_models_handle_empty_orderbook(self):
        """All models should handle empty orderbook gracefully."""
        models = [
            SimpleFillModel(),
            QueueFillModel(),
            ImpactFillModel(),
        ]

        ob = KalshiOrderbook(
            market_ticker="TEST",
            timestamp=datetime.utcnow(),
            yes_bids=[],
            yes_asks=[],
        )

        for model in models:
            fill = model.simulate_fill(ob, Side.BUY, quantity=10, limit_price=50)
            assert fill is None
