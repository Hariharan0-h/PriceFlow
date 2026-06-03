"""
Naive baseline agents — Issue #5.
All agents expose a single .act(obs) -> int interface.
"""
import numpy as np


class FixedPriceAgent:
    """Always selects the same price tier."""

    def __init__(self, price_index: int = 2):
        self.price_index = price_index

    def act(self, obs: np.ndarray) -> int:
        return self.price_index


class DailyDiscountAgent:
    """
    Starts at the highest tier and drops one tier every N days.
    Implements a ~10% daily discount by stepping down price tiers.
    """

    def __init__(self, n_price_levels: int = 7, max_days: int = 30):
        self.n_price_levels = n_price_levels
        self.max_days = max_days

    def act(self, obs: np.ndarray) -> int:
        days_remaining = int(obs[1])
        # Step down one tier for each day elapsed
        elapsed = self.max_days - days_remaining
        tier = self.n_price_levels - 1 - min(elapsed, self.n_price_levels - 1)
        return max(tier, 0)


class DemandBasedAgent:
    """
    Static demand-aware agent: lower price when inventory is high,
    raise when inventory is scarce relative to time left.
    """

    def __init__(self, n_price_levels: int = 7, max_inventory: int = 50):
        self.n_price_levels = n_price_levels
        self.max_inventory = max_inventory

    def act(self, obs: np.ndarray) -> int:
        inventory, days_remaining = int(obs[0]), int(obs[1])

        if days_remaining == 0:
            return 0  # last day — sell cheap

        inventory_ratio = inventory / self.max_inventory
        # Higher inventory → lower price tier; lower inventory → higher price
        tier = int((1.0 - inventory_ratio) * (self.n_price_levels - 1))
        return np.clip(tier, 0, self.n_price_levels - 1)
