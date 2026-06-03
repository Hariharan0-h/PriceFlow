"""
Stochastic demand function — Issue #2.
P(purchase) = f(price, days_remaining) + Gaussian noise, clipped to [0, 1].
"""
import numpy as np


def demand_probability(
    price: float,
    days_remaining: int,
    base_sensitivity: float = 0.005,
    time_boost_max: float = 0.2,
    noise_std: float = 0.05,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Returns purchase probability for a given price and time-to-departure.

    Properties guaranteed:
    - Monotonically decreasing with price (higher price → lower demand)
    - Increases near deadline (urgency effect)
    - Adds Gaussian noise for stochasticity
    """
    rng = rng or np.random.default_rng()

    # Base probability: exponential decay in price
    base = np.exp(-base_sensitivity * price)

    # Urgency boost: linearly stronger as days_remaining → 0
    max_days = 30  # assumed episode length; adjust if env changes
    urgency = time_boost_max * (1.0 - days_remaining / max_days)

    noise = rng.normal(0.0, noise_std)

    return float(np.clip(base + urgency + noise, 0.0, 1.0))
