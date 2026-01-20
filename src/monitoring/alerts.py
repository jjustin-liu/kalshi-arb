"""
Alert System for Trading Monitoring.

Generates alerts when conditions require attention:
- Toxicity thresholds exceeded
- Drawdown limits approached
- Position limits approaching
- Spread widening
- System errors

Alert levels:
- INFO: Informational, no action needed
- WARNING: Attention needed, may require action
- CRITICAL: Immediate action required
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Optional
import logging

from .config import AlertConfig

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    TOXICITY = "toxicity"
    DRAWDOWN = "drawdown"
    SPREAD = "spread"
    POSITION = "position"
    LOSS = "loss"
    SYSTEM = "system"
    CONNECTION = "connection"


@dataclass
class Alert:
    """A single alert."""
    timestamp: datetime
    level: AlertLevel
    alert_type: AlertType
    message: str
    market: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "type": self.alert_type.value,
            "message": self.message,
            "market": self.market,
            "value": self.value,
            "threshold": self.threshold,
        }


class AlertManager:
    """
    Manages alert generation and delivery.

    Monitors trading metrics and generates alerts when thresholds
    are exceeded. Includes cooldown to prevent alert spam.

    Usage:
        alerts = AlertManager(config)
        alerts.on_alert(lambda a: print(f"ALERT: {a.message}"))

        # Check metrics
        alerts.check_toxicity("INXD-24JAN15-B5850", 0.75)
        alerts.check_drawdown(0.04)
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize alert manager.

        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()

        # Alert history
        self._alerts: deque[Alert] = deque(maxlen=500)

        # Cooldown tracking (alert_type:market -> last_alert_time)
        self._cooldowns: dict[str, datetime] = {}

        # Callbacks
        self._callbacks: list[Callable[[Alert], None]] = []

    def on_alert(self, callback: Callable[[Alert], None]):
        """Register callback for alerts."""
        self._callbacks.append(callback)

    def _create_alert(
        self,
        level: AlertLevel,
        alert_type: AlertType,
        message: str,
        market: Optional[str] = None,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> Optional[Alert]:
        """Create and dispatch alert if not in cooldown."""
        # Check cooldown
        cooldown_key = f"{alert_type.value}:{market or 'global'}"
        last_alert = self._cooldowns.get(cooldown_key)

        if last_alert:
            elapsed = (datetime.utcnow() - last_alert).total_seconds()
            if elapsed < self.config.alert_cooldown_seconds:
                return None

        # Create alert
        alert = Alert(
            timestamp=datetime.utcnow(),
            level=level,
            alert_type=alert_type,
            message=message,
            market=market,
            value=value,
            threshold=threshold,
        )

        # Record
        self._alerts.append(alert)
        self._cooldowns[cooldown_key] = datetime.utcnow()

        # Log
        log_method = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }.get(level, logger.info)
        log_method(f"ALERT [{level.value.upper()}] {message}")

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        return alert

    def check_toxicity(self, market: str, score: float) -> Optional[Alert]:
        """Check toxicity score and generate alert if needed."""
        if score >= self.config.toxicity_critical:
            return self._create_alert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.TOXICITY,
                message=f"Critical toxicity in {market}: {score:.2f}",
                market=market,
                value=score,
                threshold=self.config.toxicity_critical,
            )
        elif score >= self.config.toxicity_warning:
            return self._create_alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.TOXICITY,
                message=f"Elevated toxicity in {market}: {score:.2f}",
                market=market,
                value=score,
                threshold=self.config.toxicity_warning,
            )
        return None

    def check_drawdown(self, drawdown_pct: float) -> Optional[Alert]:
        """Check drawdown and generate alert if needed."""
        if drawdown_pct >= self.config.drawdown_critical:
            return self._create_alert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.DRAWDOWN,
                message=f"Critical drawdown: {drawdown_pct:.1%}",
                value=drawdown_pct,
                threshold=self.config.drawdown_critical,
            )
        elif drawdown_pct >= self.config.drawdown_warning:
            return self._create_alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.DRAWDOWN,
                message=f"Elevated drawdown: {drawdown_pct:.1%}",
                value=drawdown_pct,
                threshold=self.config.drawdown_warning,
            )
        return None

    def check_spread(self, market: str, spread_cents: int) -> Optional[Alert]:
        """Check spread and generate alert if needed."""
        if spread_cents >= self.config.spread_critical_cents:
            return self._create_alert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.SPREAD,
                message=f"Wide spread in {market}: {spread_cents}c",
                market=market,
                value=float(spread_cents),
                threshold=float(self.config.spread_critical_cents),
            )
        elif spread_cents >= self.config.spread_warning_cents:
            return self._create_alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.SPREAD,
                message=f"Spread widening in {market}: {spread_cents}c",
                market=market,
                value=float(spread_cents),
                threshold=float(self.config.spread_warning_cents),
            )
        return None

    def check_position(
        self,
        position: int,
        limit: int,
        market: Optional[str] = None,
    ) -> Optional[Alert]:
        """Check position against limit and generate alert if needed."""
        usage = position / limit if limit > 0 else 0

        if usage >= self.config.position_critical_pct:
            return self._create_alert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.POSITION,
                message=f"Position near limit: {position}/{limit} ({usage:.0%})",
                market=market,
                value=float(position),
                threshold=float(limit * self.config.position_critical_pct),
            )
        elif usage >= self.config.position_warning_pct:
            return self._create_alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.POSITION,
                message=f"Position elevated: {position}/{limit} ({usage:.0%})",
                market=market,
                value=float(position),
                threshold=float(limit * self.config.position_warning_pct),
            )
        return None

    def check_daily_loss(self, loss: float) -> Optional[Alert]:
        """Check daily loss and generate alert if needed."""
        if loss >= self.config.daily_loss_critical:
            return self._create_alert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.LOSS,
                message=f"Critical daily loss: ${loss:.2f}",
                value=loss,
                threshold=self.config.daily_loss_critical,
            )
        elif loss >= self.config.daily_loss_warning:
            return self._create_alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.LOSS,
                message=f"Elevated daily loss: ${loss:.2f}",
                value=loss,
                threshold=self.config.daily_loss_warning,
            )
        return None

    def system_alert(self, message: str, level: AlertLevel = AlertLevel.WARNING) -> Alert:
        """Generate system alert."""
        return self._create_alert(
            level=level,
            alert_type=AlertType.SYSTEM,
            message=message,
        )

    def connection_alert(
        self,
        service: str,
        connected: bool,
    ) -> Optional[Alert]:
        """Generate connection status alert."""
        if not connected:
            return self._create_alert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.CONNECTION,
                message=f"Connection lost: {service}",
            )
        return None

    def get_recent_alerts(self, count: int = 50) -> list[Alert]:
        """Get recent alerts."""
        return list(self._alerts)[-count:]

    def get_alerts_by_level(self, level: AlertLevel) -> list[Alert]:
        """Get alerts filtered by level."""
        return [a for a in self._alerts if a.level == level]

    def get_active_critical(self) -> list[Alert]:
        """Get active critical alerts (within cooldown period)."""
        cutoff = datetime.utcnow() - timedelta(seconds=self.config.alert_cooldown_seconds * 2)
        return [
            a for a in self._alerts
            if a.level == AlertLevel.CRITICAL and a.timestamp > cutoff
        ]

    def clear_cooldowns(self):
        """Clear all cooldowns."""
        self._cooldowns.clear()
