"""
Tabular Q-learning agent — Issue #6.
Q-table shape: [inventory_bins × time_bins × price_actions]
"""
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_price_levels: int = 7,
        inventory_bins: int = 10,
        time_bins: int = 10,
        max_inventory: int = 50,
        max_days: int = 30,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_price_levels
        self.inventory_bins = inventory_bins
        self.time_bins = time_bins
        self.max_inventory = max_inventory
        self.max_days = max_days
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((inventory_bins, time_bins, n_price_levels))

    # ------------------------------------------------------------------
    def _discretize(self, obs: np.ndarray) -> tuple[int, int]:
        inv_bin = min(
            int(obs[0] / self.max_inventory * self.inventory_bins),
            self.inventory_bins - 1,
        )
        time_bin = min(
            int(obs[1] / self.max_days * self.time_bins),
            self.time_bins - 1,
        )
        return inv_bin, time_bin

    def act(self, obs: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        inv_bin, time_bin = self._discretize(obs)
        return int(np.argmax(self.q_table[inv_bin, time_bin]))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        inv_bin, time_bin = self._discretize(obs)
        n_inv_bin, n_time_bin = self._discretize(next_obs)

        current_q = self.q_table[inv_bin, time_bin, action]
        target = reward + (0 if done else self.gamma * np.max(self.q_table[n_inv_bin, n_time_bin]))
        self.q_table[inv_bin, time_bin, action] += self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
