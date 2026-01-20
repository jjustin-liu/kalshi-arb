"""Execution module configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionConfig:
    """
    Configuration for execution engine.

    Controls how orders are executed, including liquidity checks,
    timeouts, and execution strategies.
    """

    # Liquidity requirements
    min_depth: int = 10  # Minimum contracts at best price
    max_spread_cents: int = 5  # Maximum bid-ask spread
    max_impact_pct: float = 0.01  # Max expected market impact (1%)

    # Order timeouts
    fill_timeout_seconds: float = 30.0  # Cancel unfilled orders after this
    partial_fill_timeout_seconds: float = 60.0  # Cancel partial fills after this

    # Execution behavior
    use_aggressive_pricing: bool = False  # Cross spread vs post limit
    allow_partial_fills: bool = True
    default_strategy: str = "limit"  # "limit", "twap", "iceberg"

    # Rate limiting
    max_orders_per_minute: int = 60
    min_order_interval_seconds: float = 0.5

    # Position sizing
    max_pct_of_depth: float = 0.50  # Don't take more than 50% of top level


@dataclass
class TWAPConfig:
    """
    Configuration for TWAP (Time-Weighted Average Price) execution.

    TWAP splits a large order into smaller slices executed over time
    to minimize market impact and information leakage.
    """

    # Slice configuration
    num_slices: int = 5  # Number of order slices
    duration_seconds: float = 60.0  # Total execution duration
    randomize_timing: bool = True  # Add random jitter to slice timing
    timing_jitter_pct: float = 0.20  # +/- 20% timing variation

    # Slice sizing
    randomize_size: bool = True  # Vary slice sizes
    size_jitter_pct: float = 0.15  # +/- 15% size variation

    # Execution
    aggressive_final_slice: bool = True  # Cross spread for last slice
    cancel_on_adverse_move: float = 0.02  # Cancel if price moves 2% against us


@dataclass
class IcebergConfig:
    """
    Configuration for Iceberg order execution.

    Iceberg orders show only a small "visible" quantity while
    hiding the true order size. When the visible portion fills,
    more is automatically shown.
    """

    # Display configuration
    visible_quantity: int = 10  # Contracts shown at a time
    min_visible: int = 5  # Minimum visible quantity
    randomize_visible: bool = True  # Vary visible size
    visible_jitter_pct: float = 0.20  # +/- 20% visible size variation

    # Refresh behavior
    refresh_delay_seconds: float = 0.5  # Delay before showing more
    randomize_refresh: bool = True  # Vary refresh timing

    # Pricing
    price_adjustment: int = 0  # Cents to adjust from best price
    reprice_on_market_move: bool = True  # Adjust price if market moves


@dataclass
class HedgeConfig:
    """
    Configuration for delta hedging with ES futures.

    NOTE: This is simulated - no actual broker integration.
    Used for P&L attribution and risk tracking.
    """

    # Hedge parameters
    enabled: bool = True
    hedge_ratio: float = 1.0  # Delta hedge ratio (1.0 = full hedge)

    # ES futures specs
    es_contract_multiplier: float = 50.0  # $50 per point
    es_tick_size: float = 0.25  # Minimum price increment

    # Simulation
    simulated_slippage_ticks: float = 0.5  # Assumed slippage per trade
    simulated_commission: float = 2.50  # Per contract per side

    # Hedge thresholds
    min_delta_to_hedge: float = 5.0  # Minimum delta before hedging
    rehedge_threshold: float = 10.0  # Rehedge when delta exceeds this
