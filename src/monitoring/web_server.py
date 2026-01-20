"""
FastAPI Web Server for Monitoring Dashboard.

Provides:
- REST API for state, metrics, alerts
- WebSocket for real-time updates
- Prometheus metrics endpoint
- Static HTML dashboard
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .config import DashboardConfig
from .dashboard import DashboardManager
from .alerts import AlertManager, Alert
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class MonitoringServer:
    """
    FastAPI-based monitoring server.

    Endpoints:
    - GET / : HTML dashboard
    - GET /api/state : Current system state
    - GET /api/markets : Market data
    - GET /api/pnl : P&L data
    - GET /api/alerts : Recent alerts
    - GET /metrics : Prometheus metrics
    - WS /ws : Real-time WebSocket updates

    Usage:
        server = MonitoringServer(dashboard, alerts, metrics)
        await server.start()
        # ... later ...
        await server.stop()
    """

    def __init__(
        self,
        dashboard: DashboardManager,
        alerts: AlertManager,
        metrics: MetricsCollector,
        config: Optional[DashboardConfig] = None,
    ):
        """
        Initialize monitoring server.

        Args:
            dashboard: Dashboard state manager
            alerts: Alert manager
            metrics: Prometheus metrics collector
            config: Server configuration
        """
        self.dashboard = dashboard
        self.alerts = alerts
        self.metrics = metrics
        self.config = config or DashboardConfig()

        # FastAPI app
        self.app = FastAPI(title="Kalshi Arbitrage Monitor")

        # WebSocket connections
        self._ws_clients: set[WebSocket] = set()

        # Setup routes
        self._setup_routes()

        # Register dashboard callback for WebSocket updates
        dashboard.on_update(self._on_state_update)

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """Serve HTML dashboard."""
            return self._generate_dashboard_html()

        @self.app.get("/api/state")
        async def get_state():
            """Get current system state."""
            return JSONResponse(self.dashboard.to_dict())

        @self.app.get("/api/markets")
        async def get_markets():
            """Get market data."""
            state = self.dashboard.get_state()
            return JSONResponse({
                "markets": {
                    k: {
                        "ticker": v.ticker,
                        "best_bid": v.best_bid,
                        "best_ask": v.best_ask,
                        "spread": v.spread,
                        "mid_price": v.mid_price,
                        "toxicity_score": v.toxicity_score,
                    }
                    for k, v in state.markets.items()
                }
            })

        @self.app.get("/api/pnl")
        async def get_pnl():
            """Get P&L data."""
            state = self.dashboard.get_state()
            return JSONResponse({
                "realized": state.realized_pnl,
                "unrealized": state.unrealized_pnl,
                "total": state.total_pnl,
                "daily": state.daily_pnl,
                "equity": state.current_equity,
                "peak_equity": state.peak_equity,
                "drawdown_pct": state.drawdown_pct,
            })

        @self.app.get("/api/positions")
        async def get_positions():
            """Get current positions."""
            state = self.dashboard.get_state()
            return JSONResponse({
                "total_position": state.total_position,
                "positions": {
                    k: {
                        "market": v.market,
                        "quantity": v.quantity,
                        "avg_entry": v.avg_entry,
                        "unrealized_pnl": v.unrealized_pnl,
                    }
                    for k, v in state.positions.items()
                }
            })

        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get recent alerts."""
            alerts = self.alerts.get_recent_alerts(50)
            return JSONResponse({
                "alerts": [a.to_dict() for a in alerts]
            })

        @self.app.get("/api/signals")
        async def get_signals():
            """Get recent signals."""
            state = self.dashboard.get_state()
            return JSONResponse({
                "signals": state.recent_signals[-50:]
            })

        @self.app.get("/api/trades")
        async def get_trades():
            """Get recent trades."""
            state = self.dashboard.get_state()
            return JSONResponse({
                "trades": state.recent_trades[-50:]
            })

        @self.app.get("/metrics")
        async def get_metrics():
            """Prometheus metrics endpoint."""
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates."""
            await websocket.accept()
            self._ws_clients.add(websocket)
            logger.info(f"WebSocket client connected. Total: {len(self._ws_clients)}")

            try:
                # Send initial state
                await websocket.send_json(self.dashboard.to_dict())

                # Keep connection alive
                while True:
                    try:
                        # Wait for messages (ping/pong)
                        data = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=self.config.ws_heartbeat_interval,
                        )
                        if data == "ping":
                            await websocket.send_text("pong")
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        await websocket.send_json({"type": "heartbeat"})

            except WebSocketDisconnect:
                pass
            finally:
                self._ws_clients.discard(websocket)
                logger.info(f"WebSocket client disconnected. Total: {len(self._ws_clients)}")

    def _on_state_update(self, state):
        """Handle state update from dashboard."""
        # Broadcast to all WebSocket clients
        asyncio.create_task(self._broadcast_state())

    async def _broadcast_state(self):
        """Broadcast state to all WebSocket clients."""
        if not self._ws_clients:
            return

        state_json = self.dashboard.to_dict()

        disconnected = set()
        for ws in self._ws_clients:
            try:
                await ws.send_json(state_json)
            except Exception:
                disconnected.add(ws)

        self._ws_clients -= disconnected

    def _generate_dashboard_html(self) -> str:
        """Generate HTML dashboard."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Kalshi Arbitrage Monitor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        h1 { color: #00d4ff; }
        .status {
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        .status.running { background: #28a745; }
        .status.stopped { background: #dc3545; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
        }
        .card h2 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 14px;
            text-transform: uppercase;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #2a3f5f;
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #888; }
        .metric-value { font-weight: bold; font-size: 18px; }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .warning { color: #ffc107; }
        .pnl-large {
            font-size: 32px;
            text-align: center;
            padding: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #2a3f5f;
        }
        th { color: #888; font-weight: normal; }
        .alert-item {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .alert-item.critical { background: rgba(220, 53, 69, 0.2); border-left: 3px solid #dc3545; }
        .alert-item.warning { background: rgba(255, 193, 7, 0.2); border-left: 3px solid #ffc107; }
        .alert-item.info { background: rgba(0, 212, 255, 0.2); border-left: 3px solid #00d4ff; }
        .toxicity-bar {
            height: 8px;
            background: #2a3f5f;
            border-radius: 4px;
            overflow: hidden;
        }
        .toxicity-fill {
            height: 100%;
            transition: width 0.3s;
        }
        .toxicity-fill.low { background: #28a745; }
        .toxicity-fill.medium { background: #ffc107; }
        .toxicity-fill.high { background: #dc3545; }
        #connection-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
        }
        #connection-status.connected { background: #28a745; }
        #connection-status.disconnected { background: #dc3545; }
    </style>
</head>
<body>
    <div id="connection-status" class="disconnected">Disconnected</div>

    <div class="header">
        <h1>Kalshi Arbitrage Monitor</h1>
        <div id="system-status" class="status stopped">Stopped</div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>P&L</h2>
            <div id="total-pnl" class="pnl-large">$0.00</div>
            <div class="metric">
                <span class="metric-label">Realized</span>
                <span id="realized-pnl" class="metric-value">$0.00</span>
            </div>
            <div class="metric">
                <span class="metric-label">Unrealized</span>
                <span id="unrealized-pnl" class="metric-value">$0.00</span>
            </div>
            <div class="metric">
                <span class="metric-label">Daily</span>
                <span id="daily-pnl" class="metric-value">$0.00</span>
            </div>
            <div class="metric">
                <span class="metric-label">Drawdown</span>
                <span id="drawdown" class="metric-value">0.0%</span>
            </div>
        </div>

        <div class="card">
            <h2>Statistics</h2>
            <div class="metric">
                <span class="metric-label">Signals Generated</span>
                <span id="signals-generated" class="metric-value">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Signals Traded</span>
                <span id="signals-traded" class="metric-value">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Signals Filtered</span>
                <span id="signals-filtered" class="metric-value">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Trades</span>
                <span id="total-trades" class="metric-value">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Volume</span>
                <span id="total-volume" class="metric-value">0</span>
            </div>
        </div>

        <div class="card">
            <h2>Underlying</h2>
            <div class="metric">
                <span class="metric-label">ES Price</span>
                <span id="es-price" class="metric-value">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Hedge Position</span>
                <span id="hedge-position" class="metric-value">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Hedge P&L</span>
                <span id="hedge-pnl" class="metric-value">$0.00</span>
            </div>
            <div class="metric">
                <span class="metric-label">Net Delta</span>
                <span id="net-delta" class="metric-value">0</span>
            </div>
        </div>

        <div class="card">
            <h2>Positions</h2>
            <div class="metric">
                <span class="metric-label">Total Position</span>
                <span id="total-position" class="metric-value">0</span>
            </div>
            <table id="positions-table">
                <thead>
                    <tr>
                        <th>Market</th>
                        <th>Qty</th>
                        <th>Avg Entry</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>

        <div class="card" style="grid-column: span 2;">
            <h2>Recent Trades</h2>
            <table id="trades-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Market</th>
                        <th>Side</th>
                        <th>Price</th>
                        <th>Qty</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>

        <div class="card">
            <h2>Alerts</h2>
            <div id="alerts-container"></div>
        </div>
    </div>

    <script>
        let ws;
        let reconnectAttempts = 0;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('connection-status').className = 'connected';
                document.getElementById('connection-status').textContent = 'Connected';
                reconnectAttempts = 0;
            };

            ws.onclose = () => {
                document.getElementById('connection-status').className = 'disconnected';
                document.getElementById('connection-status').textContent = 'Disconnected';
                setTimeout(connect, Math.min(1000 * Math.pow(2, reconnectAttempts++), 30000));
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'heartbeat') return;
                updateDashboard(data);
            };
        }

        function updateDashboard(data) {
            // System status
            const statusEl = document.getElementById('system-status');
            if (data.is_running) {
                statusEl.className = 'status running';
                statusEl.textContent = data.is_trading_enabled ? 'Trading' : 'Running';
            } else {
                statusEl.className = 'status stopped';
                statusEl.textContent = 'Stopped';
            }

            // P&L
            updatePnl('total-pnl', data.pnl.total);
            updatePnl('realized-pnl', data.pnl.realized);
            updatePnl('unrealized-pnl', data.pnl.unrealized);
            updatePnl('daily-pnl', data.pnl.daily);
            document.getElementById('drawdown').textContent = (data.pnl.drawdown_pct * 100).toFixed(1) + '%';
            document.getElementById('drawdown').className = 'metric-value ' + (data.pnl.drawdown_pct > 0.03 ? 'negative' : '');

            // Stats
            document.getElementById('signals-generated').textContent = data.stats.signals_generated;
            document.getElementById('signals-traded').textContent = data.stats.signals_traded;
            document.getElementById('signals-filtered').textContent = data.stats.signals_filtered;
            document.getElementById('total-trades').textContent = data.stats.total_trades;
            document.getElementById('total-volume').textContent = data.stats.total_volume;

            // Underlying
            document.getElementById('es-price').textContent = data.underlying.price ? data.underlying.price.toFixed(2) : '--';
            document.getElementById('hedge-position').textContent = data.hedge.position.toFixed(2);
            updatePnl('hedge-pnl', data.hedge.pnl);
            document.getElementById('net-delta').textContent = data.hedge.net_delta.toFixed(2);

            // Positions
            document.getElementById('total-position').textContent = data.stats.total_position;
            updatePositionsTable(data.positions);

            // Trades
            updateTradesTable(data.recent_trades);
        }

        function updatePnl(id, value) {
            const el = document.getElementById(id);
            el.textContent = '$' + value.toFixed(2);
            el.className = el.className.replace(/positive|negative/g, '') + ' ' + (value >= 0 ? 'positive' : 'negative');
        }

        function updatePositionsTable(positions) {
            const tbody = document.querySelector('#positions-table tbody');
            tbody.innerHTML = '';
            for (const [market, pos] of Object.entries(positions)) {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${market.substring(0, 20)}</td>
                    <td>${pos.quantity}</td>
                    <td>${pos.avg_entry.toFixed(0)}</td>
                    <td class="${pos.unrealized_pnl >= 0 ? 'positive' : 'negative'}">$${pos.unrealized_pnl.toFixed(2)}</td>
                `;
                tbody.appendChild(row);
            }
        }

        function updateTradesTable(trades) {
            const tbody = document.querySelector('#trades-table tbody');
            tbody.innerHTML = '';
            for (const trade of trades.slice(-10).reverse()) {
                const row = document.createElement('tr');
                const time = new Date(trade.timestamp).toLocaleTimeString();
                row.innerHTML = `
                    <td>${time}</td>
                    <td>${trade.market.substring(0, 15)}</td>
                    <td>${trade.side}</td>
                    <td>${trade.price}</td>
                    <td>${trade.quantity}</td>
                    <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">$${trade.pnl.toFixed(2)}</td>
                `;
                tbody.appendChild(row);
            }
        }

        // Start connection
        connect();

        // Ping to keep alive
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
            }
        }, 25000);
    </script>
</body>
</html>
"""

    async def start(self):
        """Start the server (called by uvicorn)."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()

    def run(self):
        """Run the server (blocking)."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
        )
