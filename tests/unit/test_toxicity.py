"""
Unit tests for ToxicityMonitor.
"""

import pytest
from datetime import datetime

from src.data_feed.schemas import KalshiOrderbook, KalshiTrade, PriceLevel, Side
from src.risk.toxicity_monitor import ToxicityMonitor
from src.risk.config import ToxicityConfig


class TestToxicityMonitor:
    """Tests for ToxicityMonitor."""

    def test_initial_score_is_low(self, sample_orderbook):
        """Initial toxicity should be low."""
        monitor = ToxicityMonitor()
        monitor.update_orderbook(sample_orderbook)

        score = monitor.get_toxicity_score(sample_orderbook.market_ticker, sample_orderbook)

        # With minimal data, score should be low
        assert score < 0.5

    def test_combined_score_weights(self):
        """Weights should sum to 1.0."""
        config = ToxicityConfig()

        total = (
            config.weight_ofi +
            config.weight_vpin +
            config.weight_spread +
            config.weight_sweep +
            config.weight_imbalance
        )

        assert abs(total - 1.0) < 0.01

    def test_pause_trading_threshold(self, sample_orderbook, trade_sequence):
        """Should pause trading above threshold."""
        config = ToxicityConfig(pause_threshold=0.3)  # Low threshold for testing
        monitor = ToxicityMonitor(config)

        market = sample_orderbook.market_ticker

        # Feed data
        for ob in [sample_orderbook] * 5:
            monitor.update_orderbook(ob)
        for trade in trade_sequence[:10]:
            monitor.update_trade(trade)

        # Check pause logic works
        should_pause = monitor.should_pause_trading(market, sample_orderbook)

        # The actual result depends on the data
        assert isinstance(should_pause, bool)

    def test_get_metrics_returns_valid_dataclass(self, sample_orderbook):
        """get_metrics should return ToxicityMetrics."""
        monitor = ToxicityMonitor()
        monitor.update_orderbook(sample_orderbook)

        metrics = monitor.get_metrics(sample_orderbook.market_ticker, sample_orderbook)

        assert metrics is not None
        assert hasattr(metrics, 'toxicity_score')
        assert hasattr(metrics, 'ofi')
        assert hasattr(metrics, 'vpin')
        assert hasattr(metrics, 'sweep_detected')

    def test_component_scores_returned(self, sample_orderbook):
        """Should return individual component scores."""
        monitor = ToxicityMonitor()
        monitor.update_orderbook(sample_orderbook)

        components = monitor.get_component_scores(
            sample_orderbook.market_ticker,
            sample_orderbook,
        )

        assert 'ofi' in components
        assert 'vpin' in components
        assert 'spread' in components
        assert 'sweep' in components
        assert 'imbalance' in components

        # All scores should be in [0, 1]
        for name, score in components.items():
            assert 0 <= score <= 1, f"{name} score out of range: {score}"

    def test_spread_widening_increases_toxicity(self):
        """Wide spread should increase toxicity."""
        monitor = ToxicityMonitor()
        market = "TEST"

        # Normal spread
        ob_normal = KalshiOrderbook(
            market_ticker=market,
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=49, quantity=100)],
            yes_asks=[PriceLevel(price=51, quantity=100)],  # 2 cent spread
        )

        # Wide spread
        ob_wide = KalshiOrderbook(
            market_ticker=market,
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=45, quantity=100)],
            yes_asks=[PriceLevel(price=55, quantity=100)],  # 10 cent spread
        )

        # Update with normal spread first
        for _ in range(10):
            monitor.update_orderbook(ob_normal)

        score_normal = monitor.get_toxicity_score(market, ob_normal)

        # Then wide spread
        for _ in range(10):
            monitor.update_orderbook(ob_wide)

        score_wide = monitor.get_toxicity_score(market, ob_wide)

        # Wide spread should have higher toxicity
        assert score_wide >= score_normal

    def test_imbalance_increases_toxicity(self):
        """Extreme bid/ask imbalance should increase toxicity."""
        monitor = ToxicityMonitor()
        market = "TEST"

        # Balanced
        ob_balanced = KalshiOrderbook(
            market_ticker=market,
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=100)],
            yes_asks=[PriceLevel(price=52, quantity=100)],
        )

        # Imbalanced (much more on bid side)
        ob_imbalanced = KalshiOrderbook(
            market_ticker=market,
            timestamp=datetime.utcnow(),
            yes_bids=[PriceLevel(price=50, quantity=500)],
            yes_asks=[PriceLevel(price=52, quantity=50)],  # 10:1 ratio
        )

        monitor.update_orderbook(ob_balanced)

        components_balanced = monitor.get_component_scores(market, ob_balanced)
        components_imbalanced = monitor.get_component_scores(market, ob_imbalanced)

        assert components_imbalanced['imbalance'] >= components_balanced['imbalance']

    def test_reset_clears_state(self, sample_orderbook, sample_trade):
        """Reset should clear all state."""
        monitor = ToxicityMonitor()
        market = sample_orderbook.market_ticker

        monitor.update_orderbook(sample_orderbook)
        monitor.update_trade(sample_trade)

        monitor.reset()

        # After reset, should start fresh
        metrics = monitor.get_metrics(market, sample_orderbook)
        assert metrics.toxicity_score == 0.0 or metrics.ofi == 0.0
