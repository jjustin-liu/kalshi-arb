"""
Risk Manager - Main coordinator for all risk checks.

Integrates toxicity monitoring, position limits, and loss limits
to evaluate whether signals should be traded.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from src.data_feed.schemas import (
    ArbitrageSignal,
    Fill,
    KalshiOrderbook,
    KalshiTrade,
    Side,
    ToxicityMetrics,
)

from .config import RiskConfig, RiskLimits
from .toxicity_monitor import ToxicityMonitor
from .position_limits import PositionTracker


class RejectionReason(Enum):
    """Reasons for rejecting a signal."""
    NONE = "none"
    TOXICITY_HIGH = "toxicity_high"
    POSITION_LIMIT = "position_limit"
    LOSS_LIMIT_DAILY = "loss_limit_daily"
    LOSS_LIMIT_WEEKLY = "loss_limit_weekly"
    DRAWDOWN_LIMIT = "drawdown_limit"
    ORDER_SIZE = "order_size"
    INSUFFICIENT_EDGE = "insufficient_edge"
    RISK_DISABLED = "risk_disabled"


@dataclass
class RiskAssessment:
    """
    Result of risk evaluation for a signal.

    Contains the adjusted trade parameters and any rejection reasons.
    """
    signal: ArbitrageSignal
    timestamp: datetime

    # Decision
    approved: bool
    rejection_reason: RejectionReason

    # Adjusted parameters
    adjusted_size: int  # May be reduced from recommended
    original_size: int

    # Risk metrics at time of assessment
    toxicity_score: float
    toxicity_metrics: Optional[ToxicityMetrics]
    current_position: int
    daily_pnl: float
    drawdown_pct: float

    # Warnings (non-blocking concerns)
    warnings: list[str] = field(default_factory=list)

    @property
    def size_reduced(self) -> bool:
        """Check if size was reduced from original."""
        return self.adjusted_size < self.original_size


class RiskManager:
    """
    Coordinates all risk checks for trading signals.

    Evaluates signals against:
    1. Toxicity thresholds (from OFI, VPIN, sweeps)
    2. Position limits (per market and total)
    3. Loss limits (daily, weekly, drawdown)

    Usage:
        risk_mgr = RiskManager(config, initial_capital=10000)

        # Feed market data
        risk_mgr.update_orderbook(orderbook)
        risk_mgr.update_trade(trade)

        # Evaluate signal
        assessment = risk_mgr.evaluate_signal(signal, orderbook)
        if assessment.approved:
            # Execute trade with assessment.adjusted_size
            ...
            # Record fill
            risk_mgr.record_fill(fill)
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        initial_capital: float = 10000.0,
    ):
        """
        Initialize risk manager.

        Args:
            config: Risk configuration
            initial_capital: Starting capital for P&L tracking
        """
        self.config = config or RiskConfig()

        # Components
        self.toxicity = ToxicityMonitor(self.config.toxicity)
        self.positions = PositionTracker(initial_capital)

    def update_orderbook(self, orderbook: KalshiOrderbook):
        """
        Update risk manager with new orderbook.

        Args:
            orderbook: New orderbook state
        """
        self.toxicity.update_orderbook(orderbook)

        # Update mark price for unrealized P&L
        if orderbook.mid_price is not None:
            self.positions.update_mark_prices({
                orderbook.market_ticker: orderbook.mid_price
            })

    def update_trade(self, trade: KalshiTrade):
        """
        Update risk manager with new market trade.

        Args:
            trade: Market trade (not our execution)
        """
        self.toxicity.update_trade(trade)

    def record_fill(self, fill: Fill):
        """
        Record our trade execution.

        Args:
            fill: Our trade fill
        """
        self.positions.add_fill(fill)

    def evaluate_signal(
        self,
        signal: ArbitrageSignal,
        orderbook: Optional[KalshiOrderbook] = None,
    ) -> RiskAssessment:
        """
        Evaluate a signal against all risk checks.

        Args:
            signal: Arbitrage signal to evaluate
            orderbook: Current orderbook (for toxicity)

        Returns:
            RiskAssessment with approval decision and adjusted parameters
        """
        market = signal.market_ticker
        warnings = []
        original_size = signal.recommended_size
        adjusted_size = original_size

        # Check if risk checks are enabled
        if not self.config.enabled:
            return RiskAssessment(
                signal=signal,
                timestamp=datetime.utcnow(),
                approved=True if self.config.paper_trading else False,
                rejection_reason=RejectionReason.RISK_DISABLED,
                adjusted_size=adjusted_size,
                original_size=original_size,
                toxicity_score=0.0,
                toxicity_metrics=None,
                current_position=0,
                daily_pnl=0.0,
                drawdown_pct=0.0,
                warnings=["Risk checks disabled"],
            )

        # Get current state
        toxicity_score = self.toxicity.get_toxicity_score(market, orderbook)
        toxicity_metrics = self.toxicity.get_metrics(market, orderbook)

        pos = self.positions.get_position(market)
        current_position = pos.quantity if pos else 0
        daily_pnl = self.positions.get_daily_pnl()
        _, drawdown_pct = self.positions.get_drawdown()
        weekly_pnl = self.positions.get_weekly_pnl()
        peak_equity = self.positions._peak_equity
        current_equity = self.positions._current_equity

        # 1. Check toxicity
        if self.toxicity.should_pause_trading(market, orderbook):
            return RiskAssessment(
                signal=signal,
                timestamp=datetime.utcnow(),
                approved=False,
                rejection_reason=RejectionReason.TOXICITY_HIGH,
                adjusted_size=0,
                original_size=original_size,
                toxicity_score=toxicity_score,
                toxicity_metrics=toxicity_metrics,
                current_position=current_position,
                daily_pnl=daily_pnl,
                drawdown_pct=drawdown_pct,
            )

        # 2. Check loss limits
        limits = self.config.limits
        loss_ok, loss_reason = limits.check_loss_limit(
            daily_pnl, weekly_pnl, peak_equity, current_equity
        )

        if not loss_ok:
            reason = RejectionReason.LOSS_LIMIT_DAILY
            if "weekly" in loss_reason.lower():
                reason = RejectionReason.LOSS_LIMIT_WEEKLY
            elif "drawdown" in loss_reason.lower():
                reason = RejectionReason.DRAWDOWN_LIMIT

            return RiskAssessment(
                signal=signal,
                timestamp=datetime.utcnow(),
                approved=False,
                rejection_reason=reason,
                adjusted_size=0,
                original_size=original_size,
                toxicity_score=toxicity_score,
                toxicity_metrics=toxicity_metrics,
                current_position=current_position,
                daily_pnl=daily_pnl,
                drawdown_pct=drawdown_pct,
                warnings=[loss_reason],
            )

        # 3. Check position limits
        pos_ok, pos_reason, adjusted_size = self.positions.check_position_limit(
            market=market,
            side=signal.side,
            quantity=original_size,
            max_per_market=limits.max_position_per_market,
            max_total=limits.max_total_position,
        )

        if not pos_ok:
            return RiskAssessment(
                signal=signal,
                timestamp=datetime.utcnow(),
                approved=False,
                rejection_reason=RejectionReason.POSITION_LIMIT,
                adjusted_size=0,
                original_size=original_size,
                toxicity_score=toxicity_score,
                toxicity_metrics=toxicity_metrics,
                current_position=current_position,
                daily_pnl=daily_pnl,
                drawdown_pct=drawdown_pct,
                warnings=[pos_reason],
            )

        if pos_reason:
            warnings.append(pos_reason)

        # 4. Check order size limit
        if adjusted_size > limits.max_order_size:
            warnings.append(f"Reduced from {adjusted_size} to max order size {limits.max_order_size}")
            adjusted_size = limits.max_order_size

        # 5. Adjust size based on toxicity (gradual reduction)
        if toxicity_score > 0.3:
            # Scale down size as toxicity increases
            toxicity_reduction = 1.0 - (toxicity_score - 0.3) / 0.7
            toxicity_adjusted = int(adjusted_size * toxicity_reduction)
            if toxicity_adjusted < adjusted_size:
                warnings.append(f"Reduced from {adjusted_size} to {toxicity_adjusted} due to toxicity")
                adjusted_size = toxicity_adjusted

        # 6. Ensure minimum viable size
        if adjusted_size < 1:
            return RiskAssessment(
                signal=signal,
                timestamp=datetime.utcnow(),
                approved=False,
                rejection_reason=RejectionReason.ORDER_SIZE,
                adjusted_size=0,
                original_size=original_size,
                toxicity_score=toxicity_score,
                toxicity_metrics=toxicity_metrics,
                current_position=current_position,
                daily_pnl=daily_pnl,
                drawdown_pct=drawdown_pct,
                warnings=warnings,
            )

        # 7. Check minimum edge (signal may already check this)
        if signal.net_edge < 0.005:  # 50 bps minimum after all adjustments
            return RiskAssessment(
                signal=signal,
                timestamp=datetime.utcnow(),
                approved=False,
                rejection_reason=RejectionReason.INSUFFICIENT_EDGE,
                adjusted_size=0,
                original_size=original_size,
                toxicity_score=toxicity_score,
                toxicity_metrics=toxicity_metrics,
                current_position=current_position,
                daily_pnl=daily_pnl,
                drawdown_pct=drawdown_pct,
            )

        # All checks passed
        return RiskAssessment(
            signal=signal,
            timestamp=datetime.utcnow(),
            approved=True,
            rejection_reason=RejectionReason.NONE,
            adjusted_size=adjusted_size,
            original_size=original_size,
            toxicity_score=toxicity_score,
            toxicity_metrics=toxicity_metrics,
            current_position=current_position,
            daily_pnl=daily_pnl,
            drawdown_pct=drawdown_pct,
            warnings=warnings,
        )

    def get_toxicity_metrics(
        self,
        market: str,
        orderbook: Optional[KalshiOrderbook] = None,
    ) -> ToxicityMetrics:
        """Get current toxicity metrics for a market."""
        return self.toxicity.get_metrics(market, orderbook)

    def get_position_state(self):
        """Get current position state."""
        return self.positions.get_state()

    def can_trade(self, market: str) -> tuple[bool, str]:
        """
        Quick check if trading is allowed for a market.

        Returns:
            (can_trade, reason) tuple
        """
        if not self.config.enabled:
            return True, "Risk checks disabled"

        # Check loss limits
        limits = self.config.limits
        daily_pnl = self.positions.get_daily_pnl()
        weekly_pnl = self.positions.get_weekly_pnl()
        _, drawdown_pct = self.positions.get_drawdown()

        loss_ok, loss_reason = limits.check_loss_limit(
            daily_pnl,
            weekly_pnl,
            self.positions._peak_equity,
            self.positions._current_equity,
        )

        if not loss_ok:
            return False, loss_reason

        # Check toxicity
        if self.toxicity.should_pause_trading(market):
            score = self.toxicity.get_toxicity_score(market)
            return False, f"Toxicity too high ({score:.2f})"

        return True, ""

    def reset(self):
        """Reset all risk state."""
        self.toxicity.reset()
        self.positions.reset()
