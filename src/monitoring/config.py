"""Monitoring module configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DashboardConfig:
    """
    Configuration for the monitoring dashboard.
    """

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080

    # Update intervals (seconds)
    state_update_interval: float = 1.0
    metrics_update_interval: float = 5.0

    # Data retention
    max_trades_history: int = 1000
    max_signals_history: int = 1000
    max_alerts_history: int = 500

    # WebSocket
    ws_heartbeat_interval: float = 30.0


@dataclass
class AlertConfig:
    """
    Configuration for alert rules.
    """

    # Toxicity alerts
    toxicity_warning: float = 0.5
    toxicity_critical: float = 0.7

    # Drawdown alerts
    drawdown_warning: float = 0.03  # 3%
    drawdown_critical: float = 0.05  # 5%

    # Spread alerts
    spread_warning_cents: int = 5
    spread_critical_cents: int = 10

    # Position alerts
    position_warning_pct: float = 0.70  # 70% of limit
    position_critical_pct: float = 0.90  # 90% of limit

    # Loss alerts
    daily_loss_warning: float = 500.0
    daily_loss_critical: float = 800.0

    # Cooldown (seconds before repeat alert)
    alert_cooldown_seconds: float = 60.0
