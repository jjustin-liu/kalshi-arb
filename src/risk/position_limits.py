"""
Position and Loss Limit Enforcement.

Tracks positions across markets and enforces limits to prevent
excessive exposure and drawdowns.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from src.data_feed.schemas import Fill, Position, Side


@dataclass
class DailyPnL:
    """Daily P&L tracking."""
    date: date
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    trades_count: int = 0

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl - self.fees_paid


@dataclass
class PositionState:
    """Complete position state across all markets."""
    positions: dict[str, Position] = field(default_factory=dict)
    total_position: int = 0
    total_notional: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0


class PositionTracker:
    """
    Tracks positions and P&L across all markets.

    Provides:
    - Real-time position tracking per market
    - Daily/weekly P&L calculation
    - Peak equity and drawdown tracking
    - Position limit checks

    Usage:
        tracker = PositionTracker(initial_capital=10000)
        tracker.add_fill(fill)
        tracker.update_mark_prices({"INXD-24JAN15-B5850": 55.0})
        state = tracker.get_state()
    """

    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize position tracker.

        Args:
            initial_capital: Starting capital in dollars
        """
        self.initial_capital = initial_capital

        # Positions by market
        self._positions: dict[str, Position] = {}

        # P&L tracking
        self._daily_pnl: dict[date, DailyPnL] = {}
        self._peak_equity = initial_capital
        self._current_equity = initial_capital

        # Fill history
        self._fills: list[Fill] = []

        # Mark prices for unrealized P&L
        self._mark_prices: dict[str, float] = {}

    def add_fill(self, fill: Fill):
        """
        Process a fill and update positions.

        Args:
            fill: Trade execution
        """
        market = fill.market_ticker

        # Record fill
        self._fills.append(fill)

        # Update or create position
        if market not in self._positions:
            self._positions[market] = Position(
                market_ticker=market,
                quantity=0,
                avg_entry_price=0.0,
            )

        pos = self._positions[market]

        # Calculate P&L for closing portion
        pnl = 0.0
        old_qty = pos.quantity
        new_qty = old_qty + (fill.quantity if fill.side == Side.BUY else -fill.quantity)

        # Determine if closing
        if old_qty != 0:
            if (old_qty > 0 and fill.side == Side.SELL) or (old_qty < 0 and fill.side == Side.BUY):
                # Closing trade
                close_qty = min(abs(old_qty), fill.quantity)
                # P&L = (exit_price - entry_price) * quantity * direction
                price_diff = fill.price - pos.avg_entry_price
                if old_qty > 0:
                    pnl = price_diff * close_qty / 100  # Convert cents to dollars
                else:
                    pnl = -price_diff * close_qty / 100

                pos.realized_pnl += pnl

        # Update position
        if new_qty == 0:
            pos.quantity = 0
            pos.avg_entry_price = 0.0
        elif (old_qty >= 0 and new_qty > old_qty) or (old_qty <= 0 and new_qty < old_qty):
            # Adding to position
            if old_qty == 0:
                pos.avg_entry_price = fill.price
            else:
                # Weighted average entry
                total_cost = pos.avg_entry_price * abs(old_qty) + fill.price * fill.quantity
                pos.avg_entry_price = total_cost / abs(new_qty)
            pos.quantity = new_qty
        else:
            # Reducing position
            pos.quantity = new_qty

        # Update daily P&L
        today = datetime.utcnow().date()
        if today not in self._daily_pnl:
            self._daily_pnl[today] = DailyPnL(date=today)

        daily = self._daily_pnl[today]
        daily.realized_pnl += pnl
        daily.fees_paid += fill.fee
        daily.trades_count += 1

        # Update equity
        self._update_equity()

    def update_mark_prices(self, prices: dict[str, float]):
        """
        Update mark prices for unrealized P&L calculation.

        Args:
            prices: Dictionary of market -> mid price (0-100 scale)
        """
        self._mark_prices.update(prices)
        self._update_unrealized_pnl()
        self._update_equity()

    def _update_unrealized_pnl(self):
        """Update unrealized P&L for all positions."""
        for market, pos in self._positions.items():
            if pos.quantity == 0:
                pos.unrealized_pnl = 0.0
                continue

            mark = self._mark_prices.get(market)
            if mark is None:
                continue

            # Unrealized P&L = (mark - entry) * qty * direction
            price_diff = mark - pos.avg_entry_price
            if pos.quantity > 0:
                pos.unrealized_pnl = price_diff * abs(pos.quantity) / 100
            else:
                pos.unrealized_pnl = -price_diff * abs(pos.quantity) / 100

    def _update_equity(self):
        """Update current equity and peak."""
        # Sum all P&L
        total_realized = sum(
            p.realized_pnl for p in self._positions.values()
        )
        total_unrealized = sum(
            p.unrealized_pnl for p in self._positions.values()
        )
        total_fees = sum(
            d.fees_paid for d in self._daily_pnl.values()
        )

        self._current_equity = (
            self.initial_capital + total_realized + total_unrealized - total_fees
        )

        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity

    def get_position(self, market: str) -> Optional[Position]:
        """Get position for a specific market."""
        return self._positions.get(market)

    def get_total_position(self) -> int:
        """Get total absolute position across all markets."""
        return sum(abs(p.quantity) for p in self._positions.values())

    def get_total_notional(self) -> float:
        """Get total notional value across all positions."""
        notional = 0.0
        for market, pos in self._positions.items():
            if pos.quantity != 0:
                mark = self._mark_prices.get(market, 50)  # Default to 50
                notional += abs(pos.quantity) * mark / 100  # Convert to dollars
        return notional

    def get_daily_pnl(self, dt: Optional[date] = None) -> float:
        """Get P&L for a specific day (default today)."""
        if dt is None:
            dt = datetime.utcnow().date()

        if dt not in self._daily_pnl:
            return 0.0

        return self._daily_pnl[dt].total_pnl

    def get_weekly_pnl(self) -> float:
        """Get P&L for the current week (last 7 days)."""
        today = datetime.utcnow().date()
        total = 0.0

        for i in range(7):
            dt = date(today.year, today.month, today.day)
            if i > 0:
                from datetime import timedelta
                dt = today - timedelta(days=i)
            if dt in self._daily_pnl:
                total += self._daily_pnl[dt].total_pnl

        return total

    def get_drawdown(self) -> tuple[float, float]:
        """
        Get current drawdown.

        Returns:
            (drawdown_dollars, drawdown_percent) tuple
        """
        if self._peak_equity <= 0:
            return 0.0, 0.0

        drawdown = self._peak_equity - self._current_equity
        drawdown_pct = drawdown / self._peak_equity

        return drawdown, drawdown_pct

    def get_state(self) -> PositionState:
        """Get complete position state."""
        return PositionState(
            positions=dict(self._positions),
            total_position=self.get_total_position(),
            total_notional=self.get_total_notional(),
            daily_pnl=self.get_daily_pnl(),
            weekly_pnl=self.get_weekly_pnl(),
            peak_equity=self._peak_equity,
            current_equity=self._current_equity,
        )

    def check_position_limit(
        self,
        market: str,
        side: Side,
        quantity: int,
        max_per_market: int,
        max_total: int,
    ) -> tuple[bool, str, int]:
        """
        Check if a trade would violate position limits.

        Args:
            market: Market ticker
            side: Trade side
            quantity: Trade quantity
            max_per_market: Maximum position per market
            max_total: Maximum total position

        Returns:
            (allowed, reason, adjusted_quantity) tuple
        """
        current_pos = self._positions.get(market)
        current_qty = current_pos.quantity if current_pos else 0

        # Calculate new position
        qty_delta = quantity if side == Side.BUY else -quantity
        new_pos = current_qty + qty_delta

        # Check per-market limit
        if abs(new_pos) > max_per_market:
            # Calculate max allowed
            if side == Side.BUY:
                max_allowed = max_per_market - current_qty
            else:
                max_allowed = current_qty + max_per_market

            if max_allowed <= 0:
                return False, f"Would exceed per-market limit ({max_per_market})", 0

            return True, "Reduced to fit limit", max_allowed

        # Check total limit
        total_change = abs(new_pos) - abs(current_qty)
        new_total = self.get_total_position() + total_change

        if new_total > max_total:
            # Calculate reduction needed
            excess = new_total - max_total
            adjusted = max(0, quantity - excess)
            if adjusted == 0:
                return False, f"Would exceed total limit ({max_total})", 0
            return True, "Reduced to fit total limit", adjusted

        return True, "", quantity

    def reset(self):
        """Reset all positions and P&L."""
        self._positions.clear()
        self._daily_pnl.clear()
        self._peak_equity = self.initial_capital
        self._current_equity = self.initial_capital
        self._fills.clear()
        self._mark_prices.clear()
