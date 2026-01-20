"""
Pytest configuration and shared fixtures.
"""

import pytest
from datetime import datetime, timedelta
from typing import Generator

from src.data_feed.schemas import (
    KalshiOrderbook,
    KalshiTrade,
    KalshiMarket,
    UnderlyingTick,
    PriceLevel,
    Side,
    MarketStatus,
    ArbitrageSignal,
)


@pytest.fixture
def sample_orderbook() -> KalshiOrderbook:
    """Create a sample orderbook for testing."""
    return KalshiOrderbook(
        market_ticker="INXD-24JAN15-B5850",
        timestamp=datetime.utcnow(),
        yes_bids=[
            PriceLevel(price=55, quantity=50),
            PriceLevel(price=54, quantity=100),
            PriceLevel(price=53, quantity=150),
        ],
        yes_asks=[
            PriceLevel(price=57, quantity=40),
            PriceLevel(price=58, quantity=80),
            PriceLevel(price=59, quantity=120),
        ],
    )


@pytest.fixture
def sample_trade() -> KalshiTrade:
    """Create a sample trade for testing."""
    return KalshiTrade(
        market_ticker="INXD-24JAN15-B5850",
        timestamp=datetime.utcnow(),
        price=56,
        quantity=10,
        taker_side=Side.BUY,
    )


@pytest.fixture
def sample_market() -> KalshiMarket:
    """Create a sample market for testing."""
    return KalshiMarket(
        ticker="INXD-24JAN15-B5850",
        event_ticker="INXD-24JAN15",
        title="S&P 500 above 5850",
        strike_price=5850.0,
        expiry=datetime.utcnow() + timedelta(days=1),
        status=MarketStatus.OPEN,
    )


@pytest.fixture
def sample_underlying() -> UnderlyingTick:
    """Create a sample underlying tick for testing."""
    return UnderlyingTick(
        symbol="ES.FUT",
        timestamp=datetime.utcnow(),
        price=5865.25,
        size=1,
        bid_price=5865.00,
        bid_size=100,
        ask_price=5865.50,
        ask_size=100,
    )


@pytest.fixture
def sample_signal(sample_orderbook, sample_underlying) -> ArbitrageSignal:
    """Create a sample arbitrage signal for testing."""
    return ArbitrageSignal(
        timestamp=datetime.utcnow(),
        market_ticker="INXD-24JAN15-B5850",
        underlying_symbol="ES.FUT",
        underlying_price=5865.25,
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
        recommended_size=20,
        confidence=0.75,
    )


@pytest.fixture
def orderbook_sequence() -> list[KalshiOrderbook]:
    """Create a sequence of orderbook updates for testing."""
    base_time = datetime.utcnow()
    orderbooks = []

    for i in range(10):
        orderbooks.append(
            KalshiOrderbook(
                market_ticker="INXD-24JAN15-B5850",
                timestamp=base_time + timedelta(seconds=i),
                yes_bids=[
                    PriceLevel(price=55 + (i % 3) - 1, quantity=50 + i * 5),
                ],
                yes_asks=[
                    PriceLevel(price=57 + (i % 3) - 1, quantity=40 + i * 5),
                ],
            )
        )

    return orderbooks


@pytest.fixture
def trade_sequence() -> list[KalshiTrade]:
    """Create a sequence of trades for testing."""
    base_time = datetime.utcnow()
    trades = []

    for i in range(20):
        trades.append(
            KalshiTrade(
                market_ticker="INXD-24JAN15-B5850",
                timestamp=base_time + timedelta(seconds=i * 0.5),
                price=55 + (i % 5),
                quantity=5 + (i % 10),
                taker_side=Side.BUY if i % 3 != 0 else Side.SELL,
            )
        )

    return trades
