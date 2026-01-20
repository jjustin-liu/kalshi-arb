"""Risk module configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskLimits:
    """
    Position and loss limits for risk management.

    These limits prevent excessive exposure and drawdowns.
    """

    # Position limits
    max_position_per_market: int = 100  # Max contracts per market
    max_total_position: int = 500  # Max contracts across all markets
    max_notional_per_market: float = 5000.0  # Max $ notional per market
    max_total_notional: float = 25000.0  # Max $ notional total

    # Loss limits
    max_daily_loss: float = 1000.0  # Max daily P&L loss
    max_weekly_loss: float = 3000.0  # Max weekly P&L loss
    max_drawdown_pct: float = 0.05  # 5% max drawdown from peak

    # Order limits
    max_order_size: int = 50  # Max contracts per order
    max_orders_per_minute: int = 60  # Rate limit

    # Concentration limits
    max_position_pct_of_oi: float = 0.10  # Max 10% of open interest

    def check_position_limit(
        self,
        current_position: int,
        new_size: int,
        market_ticker: str,
    ) -> tuple[bool, str]:
        """
        Check if a new trade would violate position limits.

        Returns:
            (allowed, reason) tuple
        """
        new_position = current_position + new_size

        if abs(new_position) > self.max_position_per_market:
            return False, f"Exceeds max position per market ({self.max_position_per_market})"

        if abs(new_size) > self.max_order_size:
            return False, f"Exceeds max order size ({self.max_order_size})"

        return True, ""

    def check_loss_limit(
        self,
        daily_pnl: float,
        weekly_pnl: float,
        peak_equity: float,
        current_equity: float,
    ) -> tuple[bool, str]:
        """
        Check if loss limits have been breached.

        Returns:
            (within_limits, reason) tuple
        """
        if daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss limit breached (${-daily_pnl:.2f} > ${self.max_daily_loss:.2f})"

        if weekly_pnl < -self.max_weekly_loss:
            return False, f"Weekly loss limit breached (${-weekly_pnl:.2f} > ${self.max_weekly_loss:.2f})"

        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity
            if drawdown > self.max_drawdown_pct:
                return False, f"Drawdown limit breached ({drawdown:.1%} > {self.max_drawdown_pct:.1%})"

        return True, ""


@dataclass
class ToxicityConfig:
    """
    Configuration for toxicity detection.

    Toxicity measures how likely it is that adverse selection is present -
    i.e., informed traders picking off stale quotes.
    """

    # OFI (Order Flow Imbalance) settings
    ofi_window_seconds: float = 60.0  # Lookback window for OFI
    ofi_zscore_threshold: float = 2.0  # Z-score above which OFI is concerning

    # VPIN (Volume-Synchronized PIN) settings
    vpin_bucket_size: int = 100  # Contracts per bucket
    vpin_num_buckets: int = 50  # Number of rolling buckets
    vpin_threshold: float = 0.6  # VPIN above this is toxic

    # Sweep detection
    sweep_time_window_ms: int = 500  # Time window to detect sweep
    sweep_min_levels: int = 2  # Min price levels cleared
    sweep_cooldown_seconds: float = 5.0  # Cooldown after sweep detected

    # Spread monitoring
    spread_zscore_threshold: float = 2.0  # Widening spread = toxicity

    # Imbalance settings
    imbalance_ratio_threshold: float = 3.0  # Bid/ask imbalance > 3x

    # Combined score weights (must sum to 1.0)
    weight_ofi: float = 0.25
    weight_vpin: float = 0.30
    weight_spread: float = 0.20
    weight_sweep: float = 0.15
    weight_imbalance: float = 0.10

    # Trading pause threshold
    pause_threshold: float = 0.6  # Pause trading above this toxicity

    def __post_init__(self):
        """Validate weights sum to 1."""
        total = (
            self.weight_ofi +
            self.weight_vpin +
            self.weight_spread +
            self.weight_sweep +
            self.weight_imbalance
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Toxicity weights must sum to 1.0, got {total}")


@dataclass
class RiskConfig:
    """Combined risk configuration."""

    limits: RiskLimits = field(default_factory=RiskLimits)
    toxicity: ToxicityConfig = field(default_factory=ToxicityConfig)

    # Global settings
    enabled: bool = True  # Master switch for risk checks
    paper_trading: bool = True  # If true, don't actually reject signals
