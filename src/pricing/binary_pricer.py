"""Binary option pricer using Black-Scholes for digital options."""

import math
from dataclasses import dataclass
from typing import Optional

from scipy.stats import norm


@dataclass
class BinaryOptionPrice:
    """Result of binary option pricing."""
    probability: float  # 0-1 probability of finishing ITM
    delta: float  # dP/dS
    gamma: float  # d2P/dS2
    vega: float  # dP/dσ (per 1% vol move)
    theta: float  # dP/dT (per day)

    # Inputs used
    spot: float
    strike: float
    volatility: float
    time_to_expiry: float
    risk_free_rate: float
    dividend_yield: float


class BinaryOptionPricer:
    """
    Prices binary/digital options using Black-Scholes framework.

    For Kalshi S&P 500 index markets (INXD), we model them as
    cash-or-nothing digital call options:
    - Pays $1 if S_T >= K (YES)
    - Pays $0 if S_T < K (NO)

    Fair probability P(S_T >= K) = e^(-rT) * N(d2)
    where d2 = (ln(S/K) + (r - q - 0.5σ²)T) / (σ√T)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.015,
    ):
        """
        Initialize pricer with market parameters.

        Args:
            risk_free_rate: Annualized risk-free rate (default 5%)
            dividend_yield: Annualized dividend yield (default 1.5% for SPX)
        """
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

    def price(
        self,
        spot: float,
        strike: float,
        volatility: float,
        time_to_expiry: float,
        risk_free_rate: Optional[float] = None,
        dividend_yield: Optional[float] = None,
    ) -> BinaryOptionPrice:
        """
        Price a binary call option (probability of spot >= strike at expiry).

        Args:
            spot: Current underlying price (e.g., ES futures price)
            strike: Strike price (Kalshi market threshold)
            volatility: Annualized volatility (e.g., 0.15 for 15%)
            time_to_expiry: Time to expiry in years
            risk_free_rate: Override default risk-free rate
            dividend_yield: Override default dividend yield

        Returns:
            BinaryOptionPrice with probability and greeks
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        q = dividend_yield if dividend_yield is not None else self.dividend_yield

        # Handle edge cases
        if time_to_expiry <= 0:
            # At/past expiry - binary outcome
            prob = 1.0 if spot >= strike else 0.0
            return BinaryOptionPrice(
                probability=prob,
                delta=0.0,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                spot=spot,
                strike=strike,
                volatility=volatility,
                time_to_expiry=0.0,
                risk_free_rate=r,
                dividend_yield=q,
            )

        if volatility <= 0:
            # Zero vol - deterministic outcome
            forward = spot * math.exp((r - q) * time_to_expiry)
            prob = 1.0 if forward >= strike else 0.0
            return BinaryOptionPrice(
                probability=prob,
                delta=0.0,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                spot=spot,
                strike=strike,
                volatility=0.0,
                time_to_expiry=time_to_expiry,
                risk_free_rate=r,
                dividend_yield=q,
            )

        # Calculate d1 and d2
        sqrt_t = math.sqrt(time_to_expiry)
        d1 = (math.log(spot / strike) + (r - q + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t

        # Probability (discounted for cash-or-nothing)
        discount = math.exp(-r * time_to_expiry)
        probability = discount * norm.cdf(d2)

        # Greeks
        n_d2 = norm.pdf(d2)

        # Delta: dP/dS
        delta = discount * n_d2 / (spot * volatility * sqrt_t)

        # Gamma: d2P/dS2
        gamma = -discount * n_d2 * (1 + d2 / (volatility * sqrt_t)) / (spot**2 * volatility * sqrt_t)

        # Vega: dP/dσ (per 1% vol move, so divide by 100)
        vega = -discount * n_d2 * d1 / volatility / 100

        # Theta: dP/dT (per day, so divide by 365)
        theta_continuous = (
            r * probability +
            discount * n_d2 * (
                (r - q) / (volatility * sqrt_t) -
                d2 / (2 * time_to_expiry)
            )
        )
        theta = -theta_continuous / 365

        return BinaryOptionPrice(
            probability=probability,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            spot=spot,
            strike=strike,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            risk_free_rate=r,
            dividend_yield=q,
        )

    def implied_vol(
        self,
        spot: float,
        strike: float,
        market_probability: float,
        time_to_expiry: float,
        risk_free_rate: Optional[float] = None,
        dividend_yield: Optional[float] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> Optional[float]:
        """
        Calculate implied volatility from market probability.

        Uses Newton-Raphson iteration to find the volatility that
        produces the observed market probability.

        Args:
            spot: Current underlying price
            strike: Strike price
            market_probability: Observed probability from market (0-1)
            time_to_expiry: Time to expiry in years
            risk_free_rate: Override default risk-free rate
            dividend_yield: Override default dividend yield
            max_iterations: Maximum Newton-Raphson iterations
            tolerance: Convergence tolerance

        Returns:
            Implied volatility, or None if no solution found
        """
        if time_to_expiry <= 0:
            return None

        if market_probability <= 0 or market_probability >= 1:
            return None

        # Initial guess
        vol = 0.20  # 20% starting point

        for _ in range(max_iterations):
            result = self.price(
                spot, strike, vol, time_to_expiry,
                risk_free_rate, dividend_yield,
            )

            diff = result.probability - market_probability

            if abs(diff) < tolerance:
                return vol

            # Newton-Raphson update
            # vega is per 1% move, so multiply by 100
            if abs(result.vega) < 1e-10:
                break

            vol -= diff / (result.vega * 100)

            # Bound volatility to reasonable range
            vol = max(0.01, min(5.0, vol))

        return None

    def price_spread(
        self,
        spot: float,
        lower_strike: float,
        upper_strike: float,
        volatility: float,
        time_to_expiry: float,
    ) -> float:
        """
        Price probability of spot finishing between two strikes.

        P(lower <= S_T < upper) = P(S_T >= lower) - P(S_T >= upper)

        Args:
            spot: Current underlying price
            lower_strike: Lower bound
            upper_strike: Upper bound
            volatility: Annualized volatility
            time_to_expiry: Time to expiry in years

        Returns:
            Probability of finishing in range
        """
        p_lower = self.price(spot, lower_strike, volatility, time_to_expiry).probability
        p_upper = self.price(spot, upper_strike, volatility, time_to_expiry).probability
        return p_lower - p_upper


def calculate_moneyness(spot: float, strike: float) -> float:
    """
    Calculate log moneyness of option.

    Positive = ITM (spot > strike for call)
    Negative = OTM (spot < strike for call)
    """
    return math.log(spot / strike)


def annualize_time(seconds: float) -> float:
    """Convert seconds to annualized time."""
    return seconds / (365.25 * 24 * 3600)
