"""
Re-train DQN from scratch — Issue #15.
Usage: python train.py
"""
import os
import numpy as np

from env import DynamicPricingEnv
from agents import DQNAgent
from utils import plot_reward_curve

# ── Config ────────────────────────────────────────────────────────────
N_EPISODES = 1000
MAX_INVENTORY = 50
MAX_DAYS = 30
PRICE_LEVELS = [50, 75, 100, 125, 150, 175, 200]
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "dqn_best.pt")

# ── Setup ─────────────────────────────────────────────────────────────
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

env = DynamicPricingEnv(
    max_inventory=MAX_INVENTORY,
    max_days=MAX_DAYS,
    price_levels=PRICE_LEVELS,
)

agent = DQNAgent(
    state_dim=2,
    n_actions=len(PRICE_LEVELS),
    lr=1e-3,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    buffer_capacity=10_000,
    batch_size=64,
    target_update_freq=100,
)

# ── Training loop ─────────────────────────────────────────────────────
episode_rewards: list[float] = []
best_reward = float("-inf")

for ep in range(1, N_EPISODES + 1):
    obs, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store(obs, action, reward, next_obs, done)
        agent.learn()
        obs = next_obs
        total_reward += reward

    agent.decay_epsilon()
    episode_rewards.append(total_reward)

    if total_reward > best_reward:
        best_reward = total_reward
        agent.save(CHECKPOINT_PATH)

    if ep % 100 == 0:
        recent_mean = np.mean(episode_rewards[-100:])
        print(f"Episode {ep:4d} | Mean(last 100): {recent_mean:7.1f} | ε: {agent.epsilon:.3f}")

env.close()

# ── Save results & plot ───────────────────────────────────────────────
np.save(os.path.join(CHECKPOINT_DIR, "episode_rewards.npy"), np.array(episode_rewards))
plot_reward_curve(episode_rewards, save_path="checkpoints/reward_curve.png")
print(f"\nTraining complete. Best reward: {best_reward:.1f}")
print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
