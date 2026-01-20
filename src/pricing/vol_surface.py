"""Volatility estimation and surface modeling."""

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np


@dataclass
class VolatilityEstimate:
    """Volatility estimate with confidence."""
    volatility: float  # Annualized
    timestamp: datetime
    method: str  # "realized", "ewma", "garch", etc.
    lookback_seconds: float
    num_observations: int
    confidence: float  # 0-1


class RealizedVolatilityCalculator:
    """
    Calculates realized volatility from price data.

    Uses rolling window of log returns to estimate volatility.
    Annualizes based on trading hours.
    """

    # ES futures trade ~23 hours/day, 5.5 days/week
    TRADING_HOURS_PER_DAY = 23
    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        window_seconds: float = 3600,  # 1 hour default
        min_observations: int = 30,
        sampling_interval_seconds: float = 1.0,
    ):
        """
        Initialize volatility calculator.

        Args:
            window_seconds: Lookback window in seconds
            min_observations: Minimum observations for valid estimate
            sampling_interval_seconds: Expected time between observations
        """
        self.window_seconds = window_seconds
        self.min_observations = min_observations
        self.sampling_interval = sampling_interval_seconds

        # Store (timestamp, price) tuples
        self._prices: deque[tuple[datetime, float]] = deque()

    def update(self, timestamp: datetime, price: float):
        """Add new price observation."""
        self._prices.append((timestamp, price))
        self._prune_old(timestamp)

    def _prune_old(self, current_time: datetime):
        """Remove observations outside window."""
        cutoff = current_time - timedelta(seconds=self.window_seconds)
        while self._prices and self._prices[0][0] < cutoff:
            self._prices.popleft()

    def get_volatility(self, timestamp: Optional[datetime] = None) -> Optional[VolatilityEstimate]:
        """
        Calculate current realized volatility.

        Returns:
            VolatilityEstimate or None if insufficient data
        """
        if timestamp:
            self._prune_old(timestamp)

        if len(self._prices) < self.min_observations:
            return None

        # Calculate log returns
        prices = [p[1] for p in self._prices]
        log_returns = np.diff(np.log(prices))

        if len(log_returns) < 2:
            return None

        # Calculate volatility
        std_dev = np.std(log_returns, ddof=1)

        # Annualize
        # observations_per_year = seconds_per_year / sampling_interval
        seconds_per_year = self.TRADING_HOURS_PER_DAY * 3600 * self.TRADING_DAYS_PER_YEAR
        annualization_factor = math.sqrt(seconds_per_year / self.sampling_interval)
        annualized_vol = std_dev * annualization_factor

        current_ts = self._prices[-1][0] if self._prices else datetime.utcnow()

        return VolatilityEstimate(
            volatility=annualized_vol,
            timestamp=current_ts,
            method="realized",
            lookback_seconds=self.window_seconds,
            num_observations=len(self._prices),
            confidence=min(1.0, len(self._prices) / (self.min_observations * 2)),
        )

    def clear(self):
        """Clear price history."""
        self._prices.clear()


class EWMAVolatilityCalculator:
    """
    Exponentially weighted moving average volatility.

    More responsive to recent price changes than simple realized vol.
    """

    def __init__(
        self,
        decay_factor: float = 0.94,  # RiskMetrics uses 0.94 for daily
        min_observations: int = 20,
    ):
        """
        Initialize EWMA calculator.

        Args:
            decay_factor: Lambda parameter (0.94 typical for daily data)
            min_observations: Minimum observations for valid estimate
        """
        self.decay_factor = decay_factor
        self.min_observations = min_observations

        self._variance: Optional[float] = None
        self._last_price: Optional[float] = None
        self._last_timestamp: Optional[datetime] = None
        self._num_observations: int = 0

    def update(self, timestamp: datetime, price: float):
        """Update EWMA variance with new price."""
        if self._last_price is not None:
            log_return = math.log(price / self._last_price)

            if self._variance is None:
                # Initialize with squared return
                self._variance = log_return ** 2
            else:
                # EWMA update
                self._variance = (
                    self.decay_factor * self._variance +
                    (1 - self.decay_factor) * log_return ** 2
                )

        self._last_price = price
        self._last_timestamp = timestamp
        self._num_observations += 1

    def get_volatility(self) -> Optional[VolatilityEstimate]:
        """Get current EWMA volatility estimate."""
        if self._variance is None or self._num_observations < self.min_observations:
            return None

        # Annualize (assuming ~second-level updates)
        # This is approximate - proper annualization depends on update frequency
        TRADING_HOURS_PER_DAY = 23
        TRADING_DAYS_PER_YEAR = 252
        seconds_per_year = TRADING_HOURS_PER_DAY * 3600 * TRADING_DAYS_PER_YEAR
        annualized_vol = math.sqrt(self._variance * seconds_per_year)

        return VolatilityEstimate(
            volatility=annualized_vol,
            timestamp=self._last_timestamp or datetime.utcnow(),
            method="ewma",
            lookback_seconds=0,  # EWMA doesn't have fixed window
            num_observations=self._num_observations,
            confidence=min(1.0, self._num_observations / (self.min_observations * 2)),
        )

    def reset(self):
        """Reset calculator state."""
        self._variance = None
        self._last_price = None
        self._last_timestamp = None
        self._num_observations = 0


@dataclass
class VolatilitySurface:
    """
    Simple volatility surface for multiple strikes/expiries.

    For now, uses ATM volatility with moneyness adjustment.
    """
    atm_vol: float
    timestamp: datetime
    skew_slope: float = -0.10  # Vol decreases ~10% per 100% moneyness
    term_slope: float = 0.05  # Vol increases ~5% per year of tenor

    def get_vol(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
    ) -> float:
        """
        Get interpolated volatility for strike/expiry.

        Args:
            spot: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiry in years

        Returns:
            Interpolated volatility
        """
        # Log moneyness
        moneyness = math.log(strike / spot)

        # Moneyness adjustment (negative skew typical for equity indices)
        moneyness_adj = self.skew_slope * moneyness

        # Term structure adjustment
        term_adj = self.term_slope * time_to_expiry

        vol = self.atm_vol + moneyness_adj + term_adj

        # Floor at reasonable minimum
        return max(0.05, vol)


class VolatilityManager:
    """
    Manages volatility estimation from multiple sources.

    Combines realized vol, EWMA, and optionally implied vol from markets.
    """

    def __init__(
        self,
        realized_window: float = 3600,  # 1 hour
        ewma_decay: float = 0.94,
    ):
        self.realized_calc = RealizedVolatilityCalculator(
            window_seconds=realized_window,
        )
        self.ewma_calc = EWMAVolatilityCalculator(
            decay_factor=ewma_decay,
        )
        self._surface: Optional[VolatilitySurface] = None

    def update(self, timestamp: datetime, price: float):
        """Update all volatility estimators."""
        self.realized_calc.update(timestamp, price)
        self.ewma_calc.update(timestamp, price)

        # Update surface if we have estimates
        realized = self.realized_calc.get_volatility(timestamp)
        ewma = self.ewma_calc.get_volatility()

        if realized and ewma:
            # Blend estimates
            atm_vol = 0.6 * realized.volatility + 0.4 * ewma.volatility
            self._surface = VolatilitySurface(
                atm_vol=atm_vol,
                timestamp=timestamp,
            )
        elif realized:
            self._surface = VolatilitySurface(
                atm_vol=realized.volatility,
                timestamp=timestamp,
            )
        elif ewma:
            self._surface = VolatilitySurface(
                atm_vol=ewma.volatility,
                timestamp=timestamp,
            )

    def get_vol(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
    ) -> Optional[float]:
        """Get volatility for specific strike/expiry."""
        if self._surface is None:
            return None
        return self._surface.get_vol(spot, strike, time_to_expiry)

    def get_atm_vol(self) -> Optional[float]:
        """Get current ATM volatility."""
        if self._surface is None:
            return None
        return self._surface.atm_vol

    def get_surface(self) -> Optional[VolatilitySurface]:
        """Get current volatility surface."""
        return self._surface

    def get_realized_estimate(self) -> Optional[VolatilityEstimate]:
        """Get realized volatility estimate."""
        return self.realized_calc.get_volatility()

    def get_ewma_estimate(self) -> Optional[VolatilityEstimate]:
        """Get EWMA volatility estimate."""
        return self.ewma_calc.get_volatility()
