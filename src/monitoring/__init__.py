"""
Monitoring Module.

Web-based dashboard and metrics for the trading system.

Components:
- DashboardManager: State management for UI
- AlertManager: Alert generation and delivery
- MetricsCollector: Prometheus metrics
- MonitoringServer: FastAPI web server

Usage:
    from src.monitoring import MonitoringServer, DashboardManager, AlertManager, MetricsCollector

    dashboard = DashboardManager()
    alerts = AlertManager()
    metrics = MetricsCollector()

    server = MonitoringServer(dashboard, alerts, metrics)
    server.run()  # Starts at http://localhost:8080
"""

from .config import DashboardConfig, AlertConfig
from .dashboard import DashboardManager, DashboardState, MarketData, PositionData, SignalData, TradeData
from .alerts import AlertManager, Alert, AlertLevel, AlertType
from .metrics_collector import MetricsCollector
from .web_server import MonitoringServer

__all__ = [
    # Config
    "DashboardConfig",
    "AlertConfig",
    # Dashboard
    "DashboardManager",
    "DashboardState",
    "MarketData",
    "PositionData",
    "SignalData",
    "TradeData",
    # Alerts
    "AlertManager",
    "Alert",
    "AlertLevel",
    "AlertType",
    # Metrics
    "MetricsCollector",
    # Server
    "MonitoringServer",
]
