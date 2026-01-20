"""
Hedge Simulator - Tracks theoretical ES futures hedging position.

This is SIMULATED - no actual broker integration.
Used for:
- P&L attribution (separate Kalshi alpha from delta hedge)
- Risk tracking (net delta exposure)
- Hedge cost estimation

Delta hedging concept:
- When you buy YES on a Kalshi binary, you have positive delta
  (profit if underlying goes up)
- To hedge, you sell ES futures to neutralize delta
- This isolates the "edge" from directional market moves
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.data_feed.schemas import Fill, Side, UnderlyingTick

from .config import HedgeConfig


@dataclass
class HedgePosition:
    """Current hedge position state."""
    es_contracts: float  # Can be fractional for tracking
    avg_entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    total_commission: float
    total_slippage: float


@dataclass
class HedgeTrade:
    """Record of a hedge trade."""
    timestamp: datetime
    contracts: float  # Positive = long, negative = short
    price: float
    slippage: float
    commission: float
    reason: str  # "initial", "rehedge", "unwind"


@dataclass
class DeltaExposure:
    """Current delta exposure breakdown."""
    kalshi_delta: float  # Delta from Kalshi positions
    hedge_delta: float  # Delta from ES hedge
    net_delta: float  # Kalshi + hedge
    hedge_ratio: float  # hedge_delta / kalshi_delta


class HedgeSimulator:
    """
    Simulates delta hedging with ES futures.

    Tracks a theoretical hedge position without actual execution.
    Calculates hedge P&L from ES price movements.

    Why hedge?
    - Binary options have delta (sensitivity to underlying)
    - A long YES position profits if S&P goes up
    - By shorting ES futures, we can neutralize this exposure
    - This isolates our "edge" (mispricing) from market direction

    Usage:
        hedger = HedgeSimulator(config)

        # When Kalshi trade executes
        hedger.on_kalshi_fill(fill, delta=0.6)

        # As ES price moves
        hedger.update_es_price(tick)

        # Check P&L
        pnl = hedger.get_hedge_pnl()
    """

    def __init__(self, config: Optional[HedgeConfig] = None):
        """
        Initialize hedge simulator.

        Args:
            config: Hedge configuration
        """
        self.config = config or HedgeConfig()

        # Current state
        self._es_position: float = 0.0  # Net ES contracts (negative = short)
        self._avg_entry_price: float = 0.0
        self._last_es_price: Optional[float] = None
        self._realized_pnl: float = 0.0
        self._total_commission: float = 0.0
        self._total_slippage: float = 0.0

        # Kalshi delta tracking
        self._kalshi_delta: float = 0.0

        # Trade history
        self._trades: list[HedgeTrade] = []

    @property
    def enabled(self) -> bool:
        """Check if hedging is enabled."""
        return self.config.enabled

    def on_kalshi_fill(
        self,
        fill: Fill,
        delta: float,
        es_price: float,
    ):
        """
        Handle a Kalshi fill by adjusting hedge.

        Args:
            fill: Kalshi fill
            delta: Delta of the position (probability sensitivity to underlying)
            es_price: Current ES futures price
        """
        if not self.enabled:
            return

        # Update Kalshi delta
        # Positive delta = long underlying exposure
        delta_change = delta * fill.quantity
        if fill.side == Side.SELL:
            delta_change = -delta_change

        self._kalshi_delta += delta_change

        # Calculate required hedge
        target_hedge = -self._kalshi_delta * self.config.hedge_ratio

        # Check if we need to adjust
        hedge_change = target_hedge - self._es_position

        if abs(hedge_change) >= self.config.min_delta_to_hedge:
            self._execute_hedge(
                hedge_change,
                es_price,
                reason="initial" if self._es_position == 0 else "rehedge",
            )

    def check_rehedge(self, es_price: float) -> bool:
        """
        Check if rehedging is needed and execute if so.

        Args:
            es_price: Current ES price

        Returns:
            True if rehedge was executed
        """
        if not self.enabled:
            return False

        target_hedge = -self._kalshi_delta * self.config.hedge_ratio
        hedge_change = target_hedge - self._es_position

        if abs(hedge_change) >= self.config.rehedge_threshold:
            self._execute_hedge(hedge_change, es_price, reason="rehedge")
            return True

        return False

    def _execute_hedge(self, contracts: float, price: float, reason: str):
        """Execute a hedge trade (simulated)."""
        # Calculate slippage
        slippage_points = self.config.simulated_slippage_ticks * self.config.es_tick_size
        if contracts > 0:  # Buying
            execution_price = price + slippage_points
        else:  # Selling
            execution_price = price - slippage_points

        # Calculate commission
        commission = abs(contracts) * self.config.simulated_commission

        # Update position
        if self._es_position == 0:
            self._avg_entry_price = execution_price
        elif (self._es_position > 0 and contracts > 0) or (self._es_position < 0 and contracts < 0):
            # Adding to position
            total_cost = self._avg_entry_price * abs(self._es_position) + execution_price * abs(contracts)
            self._avg_entry_price = total_cost / (abs(self._es_position) + abs(contracts))
        else:
            # Reducing or flipping position
            close_qty = min(abs(self._es_position), abs(contracts))
            pnl_per_contract = (execution_price - self._avg_entry_price) * self.config.es_contract_multiplier
            if self._es_position < 0:
                pnl_per_contract = -pnl_per_contract

            self._realized_pnl += pnl_per_contract * close_qty

            # Handle position flip
            if abs(contracts) > abs(self._es_position):
                self._avg_entry_price = execution_price

        self._es_position += contracts
        self._total_commission += commission
        self._total_slippage += abs(slippage_points) * abs(contracts) * self.config.es_contract_multiplier

        # Record trade
        trade = HedgeTrade(
            timestamp=datetime.utcnow(),
            contracts=contracts,
            price=execution_price,
            slippage=slippage_points,
            commission=commission,
            reason=reason,
        )
        self._trades.append(trade)

    def update_es_price(self, tick: UnderlyingTick):
        """
        Update with new ES price for unrealized P&L.

        Args:
            tick: ES futures tick
        """
        self._last_es_price = tick.price

    def get_unrealized_pnl(self) -> float:
        """Get unrealized hedge P&L."""
        if self._last_es_price is None or self._es_position == 0:
            return 0.0

        price_change = self._last_es_price - self._avg_entry_price
        pnl = price_change * self._es_position * self.config.es_contract_multiplier

        return pnl

    def get_hedge_pnl(self) -> float:
        """Get total hedge P&L (realized + unrealized - costs)."""
        return (
            self._realized_pnl +
            self.get_unrealized_pnl() -
            self._total_commission -
            self._total_slippage
        )

    def get_position(self) -> HedgePosition:
        """Get current hedge position state."""
        return HedgePosition(
            es_contracts=self._es_position,
            avg_entry_price=self._avg_entry_price,
            unrealized_pnl=self.get_unrealized_pnl(),
            realized_pnl=self._realized_pnl,
            total_commission=self._total_commission,
            total_slippage=self._total_slippage,
        )

    def get_delta_exposure(self) -> DeltaExposure:
        """Get current delta exposure breakdown."""
        # ES delta is contracts * multiplier (each point = $50)
        # Normalize to same units as Kalshi delta
        hedge_delta = self._es_position * self.config.es_contract_multiplier

        return DeltaExposure(
            kalshi_delta=self._kalshi_delta,
            hedge_delta=hedge_delta,
            net_delta=self._kalshi_delta + hedge_delta,
            hedge_ratio=(
                abs(hedge_delta / self._kalshi_delta)
                if self._kalshi_delta != 0 else 0.0
            ),
        )

    def unwind_all(self, es_price: float):
        """Unwind entire hedge position."""
        if self._es_position != 0:
            self._execute_hedge(-self._es_position, es_price, reason="unwind")

    def get_trades(self) -> list[HedgeTrade]:
        """Get hedge trade history."""
        return self._trades.copy()

    def reset(self):
        """Reset all hedge state."""
        self._es_position = 0.0
        self._avg_entry_price = 0.0
        self._last_es_price = None
        self._realized_pnl = 0.0
        self._total_commission = 0.0
        self._total_slippage = 0.0
        self._kalshi_delta = 0.0
        self._trades.clear()
