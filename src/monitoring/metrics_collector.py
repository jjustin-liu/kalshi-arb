"""
Prometheus Metrics Collector.

Exposes trading metrics in Prometheus format for monitoring and alerting.

Metrics:
- kalshi_signals_total: Total signals generated
- kalshi_trades_total: Total trades executed
- kalshi_pnl_dollars: Current P&L in dollars
- kalshi_toxicity_score: Current toxicity score
- kalshi_basis_bps: Current basis in basis points
- kalshi_slippage_bps: Average slippage in basis points
- kalshi_position_contracts: Current position in contracts
"""

from prometheus_client import Counter, Gauge, Histogram, Info
from typing import Optional


class MetricsCollector:
    """
    Collects and exposes Prometheus metrics.

    Usage:
        collector = MetricsCollector()

        # Update metrics
        collector.record_signal(market, side, edge)
        collector.record_trade(market, side, price, quantity)
        collector.update_pnl(realized, unrealized)
        collector.update_toxicity(market, score)

        # Metrics are automatically exposed at /metrics endpoint
    """

    def __init__(self, prefix: str = "kalshi"):
        """
        Initialize metrics collector.

        Args:
            prefix: Prefix for all metric names
        """
        self.prefix = prefix

        # Counters
        self.signals_total = Counter(
            f"{prefix}_signals_total",
            "Total signals generated",
            ["market", "side", "tradeable"],
        )

        self.trades_total = Counter(
            f"{prefix}_trades_total",
            "Total trades executed",
            ["market", "side", "strategy"],
        )

        self.orders_total = Counter(
            f"{prefix}_orders_total",
            "Total orders submitted",
            ["market", "side", "status"],
        )

        self.rejections_total = Counter(
            f"{prefix}_rejections_total",
            "Total signal rejections",
            ["reason"],
        )

        # Gauges
        self.pnl_realized = Gauge(
            f"{prefix}_pnl_realized_dollars",
            "Realized P&L in dollars",
        )

        self.pnl_unrealized = Gauge(
            f"{prefix}_pnl_unrealized_dollars",
            "Unrealized P&L in dollars",
        )

        self.pnl_total = Gauge(
            f"{prefix}_pnl_total_dollars",
            "Total P&L in dollars",
        )

        self.equity = Gauge(
            f"{prefix}_equity_dollars",
            "Current equity in dollars",
        )

        self.drawdown = Gauge(
            f"{prefix}_drawdown_pct",
            "Current drawdown percentage",
        )

        self.toxicity_score = Gauge(
            f"{prefix}_toxicity_score",
            "Current toxicity score",
            ["market"],
        )

        self.basis_bps = Gauge(
            f"{prefix}_basis_bps",
            "Current basis in basis points",
            ["market"],
        )

        self.spread_cents = Gauge(
            f"{prefix}_spread_cents",
            "Current bid-ask spread in cents",
            ["market"],
        )

        self.position_contracts = Gauge(
            f"{prefix}_position_contracts",
            "Current position in contracts",
            ["market"],
        )

        self.total_position = Gauge(
            f"{prefix}_total_position_contracts",
            "Total position across all markets",
        )

        self.ofi = Gauge(
            f"{prefix}_ofi",
            "Order Flow Imbalance",
            ["market"],
        )

        self.vpin = Gauge(
            f"{prefix}_vpin",
            "Volume-Synchronized PIN",
            ["market"],
        )

        self.hedge_delta = Gauge(
            f"{prefix}_hedge_delta",
            "Current hedge delta exposure",
        )

        self.es_price = Gauge(
            f"{prefix}_es_price",
            "Current ES futures price",
        )

        # Histograms
        self.slippage = Histogram(
            f"{prefix}_slippage_bps",
            "Trade slippage in basis points",
            ["market", "side"],
            buckets=[0, 1, 2, 5, 10, 20, 50, 100],
        )

        self.fill_time = Histogram(
            f"{prefix}_fill_time_seconds",
            "Time to fill orders",
            ["strategy"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
        )

        self.signal_edge = Histogram(
            f"{prefix}_signal_edge_bps",
            "Signal edge in basis points",
            ["market"],
            buckets=[0, 10, 20, 50, 100, 200, 500],
        )

        # Info
        self.system_info = Info(
            f"{prefix}_system",
            "System information",
        )

    def record_signal(
        self,
        market: str,
        side: str,
        tradeable: bool,
        edge_bps: float,
    ):
        """Record a generated signal."""
        self.signals_total.labels(
            market=market,
            side=side,
            tradeable=str(tradeable).lower(),
        ).inc()

        self.signal_edge.labels(market=market).observe(edge_bps)

    def record_trade(
        self,
        market: str,
        side: str,
        quantity: int,
        strategy: str = "limit",
        slippage_bps: float = 0,
        fill_time_seconds: float = 0,
    ):
        """Record an executed trade."""
        self.trades_total.labels(
            market=market,
            side=side,
            strategy=strategy,
        ).inc()

        if slippage_bps > 0:
            self.slippage.labels(market=market, side=side).observe(slippage_bps)

        if fill_time_seconds > 0:
            self.fill_time.labels(strategy=strategy).observe(fill_time_seconds)

    def record_order(
        self,
        market: str,
        side: str,
        status: str,
    ):
        """Record an order event."""
        self.orders_total.labels(
            market=market,
            side=side,
            status=status,
        ).inc()

    def record_rejection(self, reason: str):
        """Record a signal rejection."""
        self.rejections_total.labels(reason=reason).inc()

    def update_pnl(
        self,
        realized: float,
        unrealized: float,
        equity: float,
        drawdown_pct: float,
    ):
        """Update P&L metrics."""
        self.pnl_realized.set(realized)
        self.pnl_unrealized.set(unrealized)
        self.pnl_total.set(realized + unrealized)
        self.equity.set(equity)
        self.drawdown.set(drawdown_pct)

    def update_toxicity(self, market: str, score: float):
        """Update toxicity score for a market."""
        self.toxicity_score.labels(market=market).set(score)

    def update_basis(self, market: str, basis_bps: float):
        """Update basis for a market."""
        self.basis_bps.labels(market=market).set(basis_bps)

    def update_spread(self, market: str, spread_cents: int):
        """Update spread for a market."""
        self.spread_cents.labels(market=market).set(spread_cents)

    def update_position(self, market: str, contracts: int):
        """Update position for a market."""
        self.position_contracts.labels(market=market).set(contracts)

    def update_total_position(self, contracts: int):
        """Update total position."""
        self.total_position.set(contracts)

    def update_ofi(self, market: str, ofi: float):
        """Update OFI for a market."""
        self.ofi.labels(market=market).set(ofi)

    def update_vpin(self, market: str, vpin: float):
        """Update VPIN for a market."""
        self.vpin.labels(market=market).set(vpin)

    def update_hedge(self, delta: float):
        """Update hedge delta."""
        self.hedge_delta.set(delta)

    def update_es_price(self, price: float):
        """Update ES futures price."""
        self.es_price.set(price)

    def set_system_info(self, **kwargs):
        """Set system info labels."""
        self.system_info.info(kwargs)
