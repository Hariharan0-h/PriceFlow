"""
Evaluation helpers — Issue #12.
run_evaluation() runs N seasons and returns per-season total revenue.
"""
import numpy as np

from env import DynamicPricingEnv


def run_evaluation(agent, n_seasons: int = 1000, env_kwargs: dict | None = None) -> np.ndarray:
    """
    Run `n_seasons` episodes and return total revenue per episode.
    Agent must expose .act(obs) -> int.
    """
    env = DynamicPricingEnv(**(env_kwargs or {}))
    revenues = np.zeros(n_seasons)

    for i in range(n_seasons):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        revenues[i] = total_reward

    env.close()
    return revenues
