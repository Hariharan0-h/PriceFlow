"""
Custom Gymnasium environment for RL dynamic pricing.
Issue #3 — implement step(), reset(), render()
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .demand import demand_probability


class DynamicPricingEnv(gym.Env):
    """
    MDP for finite-inventory revenue maximisation.

    State  : [remaining_inventory, days_until_departure]  (Box, int)
    Action : discrete price level index  (0 … n_price_levels-1)
    Reward : price × units_sold per timestep
    Done   : days_until_departure == 0  OR  remaining_inventory == 0
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_inventory: int = 50,
        max_days: int = 30,
        price_levels: list[float] | None = None,
    ):
        super().__init__()

        self.max_inventory = max_inventory
        self.max_days = max_days
        self.price_levels = price_levels or [50, 75, 100, 125, 150, 175, 200]
        self.n_price_levels = len(self.price_levels)

        # Observation: [remaining_inventory, days_until_departure]
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([max_inventory, max_days], dtype=np.int32),
            dtype=np.int32,
        )

        # Action: price tier index
        self.action_space = spaces.Discrete(self.n_price_levels)

        self._inventory: int = max_inventory
        self._days: int = max_days

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._inventory = self.max_inventory
        self._days = self.max_days
        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        price = self.price_levels[action]
        p_buy = demand_probability(price, self._days)
        units_sold = int(self.np_random.random() < p_buy)  # 0 or 1 per timestep
        units_sold = min(units_sold, self._inventory)

        reward = float(price * units_sold)

        self._inventory -= units_sold
        self._days -= 1

        terminated = self._days == 0 or self._inventory == 0
        return self._obs(), reward, terminated, False, {}

    def render(self):
        print(
            f"Days left: {self._days:3d} | Inventory: {self._inventory:3d}"
        )

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        return np.array([self._inventory, self._days], dtype=np.int32)
