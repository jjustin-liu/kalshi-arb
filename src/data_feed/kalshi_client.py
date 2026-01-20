"""Kalshi REST and WebSocket client."""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from typing import AsyncIterator, Callable, Optional
from urllib.parse import urlencode

import aiohttp
import websockets
from websockets.client import WebSocketClientProtocol

from .schemas import (
    KalshiMarket,
    KalshiOrderbook,
    KalshiTrade,
    MarketStatus,
    Order,
    OrderStatus,
    PriceLevel,
    Side,
)

logger = logging.getLogger(__name__)


class KalshiAuth:
    """Handles Kalshi API authentication."""

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret

    def sign_request(
        self,
        method: str,
        path: str,
        timestamp: int,
        body: str = "",
    ) -> str:
        """Generate HMAC signature for request."""
        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def get_auth_headers(
        self,
        method: str,
        path: str,
        body: str = "",
    ) -> dict[str, str]:
        """Get authentication headers for a request."""
        timestamp = int(time.time() * 1000)
        signature = self.sign_request(method, path, timestamp, body)

        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp),
            "Content-Type": "application/json",
        }


class KalshiRESTClient:
    """Kalshi REST API client."""

    DEMO_URL = "https://demo-api.kalshi.co"
    PROD_URL = "https://trading-api.kalshi.com"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        env: str = "demo",
    ):
        self.auth = KalshiAuth(api_key, api_secret)
        self.base_url = self.DEMO_URL if env == "demo" else self.PROD_URL
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
    ) -> dict:
        """Make authenticated request to Kalshi API."""
        session = await self._get_session()

        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"

        body = json.dumps(data) if data else ""
        headers = self.auth.get_auth_headers(method, path, body)

        async with session.request(
            method,
            url,
            headers=headers,
            data=body if data else None,
        ) as resp:
            if resp.status >= 400:
                error_text = await resp.text()
                logger.error(f"Kalshi API error {resp.status}: {error_text}")
                raise KalshiAPIError(resp.status, error_text)
            return await resp.json()

    async def get_markets(
        self,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> tuple[list[KalshiMarket], Optional[str]]:
        """Get list of markets."""
        params = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        resp = await self._request("GET", "/trade-api/v2/markets", params=params)

        markets = []
        for m in resp.get("markets", []):
            markets.append(self._parse_market(m))

        return markets, resp.get("cursor")

    async def get_market(self, ticker: str) -> KalshiMarket:
        """Get single market by ticker."""
        resp = await self._request("GET", f"/trade-api/v2/markets/{ticker}")
        return self._parse_market(resp["market"])

    async def get_orderbook(self, ticker: str, depth: int = 10) -> KalshiOrderbook:
        """Get market orderbook."""
        resp = await self._request(
            "GET",
            f"/trade-api/v2/markets/{ticker}/orderbook",
            params={"depth": depth},
        )

        return KalshiOrderbook(
            market_ticker=ticker,
            timestamp=datetime.utcnow(),
            yes_bids=[
                PriceLevel(price=lvl[0], quantity=lvl[1])
                for lvl in resp.get("orderbook", {}).get("yes", [])
                if len(lvl) >= 2
            ],
            yes_asks=[
                PriceLevel(price=lvl[0], quantity=lvl[1])
                for lvl in resp.get("orderbook", {}).get("no", [])
                if len(lvl) >= 2
            ],
        )

    async def get_trades(
        self,
        ticker: str,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> tuple[list[KalshiTrade], Optional[str]]:
        """Get recent trades for a market."""
        params = {"limit": limit, "ticker": ticker}
        if cursor:
            params["cursor"] = cursor

        resp = await self._request("GET", "/trade-api/v2/markets/trades", params=params)

        trades = []
        for t in resp.get("trades", []):
            trades.append(
                KalshiTrade(
                    market_ticker=ticker,
                    timestamp=datetime.fromisoformat(t["created_time"].replace("Z", "+00:00")),
                    price=t["yes_price"],
                    quantity=t["count"],
                    taker_side=Side.BUY if t["taker_side"] == "yes" else Side.SELL,
                )
            )

        return trades, resp.get("cursor")

    async def get_balance(self) -> float:
        """Get account balance in dollars."""
        resp = await self._request("GET", "/trade-api/v2/portfolio/balance")
        return resp["balance"] / 100  # Convert cents to dollars

    async def get_positions(self) -> list[dict]:
        """Get current positions."""
        resp = await self._request("GET", "/trade-api/v2/portfolio/positions")
        return resp.get("market_positions", [])

    async def create_order(self, order: Order) -> str:
        """Submit order to Kalshi."""
        data = {
            "ticker": order.market_ticker,
            "action": "buy" if order.side == Side.BUY else "sell",
            "side": "yes",
            "type": "limit",
            "yes_price": order.price,
            "count": order.quantity,
            "client_order_id": order.client_order_id,
        }

        resp = await self._request("POST", "/trade-api/v2/portfolio/orders", data=data)
        return resp["order"]["order_id"]

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            await self._request("DELETE", f"/trade-api/v2/portfolio/orders/{order_id}")
            return True
        except KalshiAPIError as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order(self, order_id: str) -> dict:
        """Get order status."""
        resp = await self._request("GET", f"/trade-api/v2/portfolio/orders/{order_id}")
        return resp["order"]

    def _parse_market(self, data: dict) -> KalshiMarket:
        """Parse market data from API response."""
        # Extract strike price from market rules if available
        strike_price = None
        if "floor_strike" in data:
            strike_price = data["floor_strike"]
        elif "cap_strike" in data:
            strike_price = data["cap_strike"]

        return KalshiMarket(
            ticker=data["ticker"],
            event_ticker=data["event_ticker"],
            title=data.get("title", ""),
            strike_price=strike_price,
            expiry=datetime.fromisoformat(data["close_time"].replace("Z", "+00:00")),
            status=MarketStatus(data["status"]),
            result=data.get("result"),
            volume=data.get("volume", 0),
            open_interest=data.get("open_interest", 0),
        )


class KalshiWebSocketClient:
    """Kalshi WebSocket client for real-time data."""

    DEMO_WS_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"
    PROD_WS_URL = "wss://trading-api.kalshi.com/trade-api/ws/v2"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        env: str = "demo",
    ):
        self.auth = KalshiAuth(api_key, api_secret)
        self.ws_url = self.DEMO_WS_URL if env == "demo" else self.PROD_WS_URL
        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._subscriptions: set[str] = set()
        self._callbacks: dict[str, list[Callable]] = {
            "orderbook": [],
            "trade": [],
            "ticker": [],
        }

    def on_orderbook(self, callback: Callable[[KalshiOrderbook], None]):
        """Register callback for orderbook updates."""
        self._callbacks["orderbook"].append(callback)

    def on_trade(self, callback: Callable[[KalshiTrade], None]):
        """Register callback for trade updates."""
        self._callbacks["trade"].append(callback)

    async def connect(self):
        """Connect to WebSocket."""
        timestamp = int(time.time() * 1000)
        signature = self.auth.sign_request("GET", "/trade-api/ws/v2", timestamp)

        headers = {
            "KALSHI-ACCESS-KEY": self.auth.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp),
        }

        self._ws = await websockets.connect(
            self.ws_url,
            additional_headers=headers,
            ping_interval=30,
            ping_timeout=10,
        )
        self._running = True
        logger.info("Connected to Kalshi WebSocket")

    async def disconnect(self):
        """Disconnect from WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def subscribe_orderbook(self, market_ticker: str):
        """Subscribe to orderbook updates for a market."""
        if not self._ws:
            raise RuntimeError("Not connected to WebSocket")

        msg = {
            "id": int(time.time() * 1000),
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_ticker": market_ticker,
            },
        }
        await self._ws.send(json.dumps(msg))
        self._subscriptions.add(f"orderbook:{market_ticker}")
        logger.info(f"Subscribed to orderbook for {market_ticker}")

    async def subscribe_trades(self, market_ticker: str):
        """Subscribe to trade updates for a market."""
        if not self._ws:
            raise RuntimeError("Not connected to WebSocket")

        msg = {
            "id": int(time.time() * 1000),
            "cmd": "subscribe",
            "params": {
                "channels": ["trade"],
                "market_ticker": market_ticker,
            },
        }
        await self._ws.send(json.dumps(msg))
        self._subscriptions.add(f"trade:{market_ticker}")
        logger.info(f"Subscribed to trades for {market_ticker}")

    async def listen(self) -> AsyncIterator[dict]:
        """Listen for WebSocket messages."""
        if not self._ws:
            raise RuntimeError("Not connected to WebSocket")

        while self._running:
            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=60)
                data = json.loads(msg)
                yield data
                await self._handle_message(data)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await self._ws.ping()
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                break

    async def _handle_message(self, data: dict):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type")

        if msg_type == "orderbook_snapshot" or msg_type == "orderbook_delta":
            orderbook = self._parse_orderbook_msg(data)
            for callback in self._callbacks["orderbook"]:
                callback(orderbook)

        elif msg_type == "trade":
            trade = self._parse_trade_msg(data)
            for callback in self._callbacks["trade"]:
                callback(trade)

    def _parse_orderbook_msg(self, data: dict) -> KalshiOrderbook:
        """Parse orderbook message."""
        msg = data.get("msg", {})
        return KalshiOrderbook(
            market_ticker=msg.get("market_ticker", ""),
            timestamp=datetime.utcnow(),
            yes_bids=[
                PriceLevel(price=lvl[0], quantity=lvl[1])
                for lvl in msg.get("yes", [])
                if len(lvl) >= 2
            ],
            yes_asks=[
                PriceLevel(price=lvl[0], quantity=lvl[1])
                for lvl in msg.get("no", [])
                if len(lvl) >= 2
            ],
        )

    def _parse_trade_msg(self, data: dict) -> KalshiTrade:
        """Parse trade message."""
        msg = data.get("msg", {})
        return KalshiTrade(
            market_ticker=msg.get("market_ticker", ""),
            timestamp=datetime.utcnow(),
            price=msg.get("yes_price", 0),
            quantity=msg.get("count", 0),
            taker_side=Side.BUY if msg.get("taker_side") == "yes" else Side.SELL,
        )


class KalshiAPIError(Exception):
    """Kalshi API error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Kalshi API error {status_code}: {message}")
