"""
Visualisation helpers — Issues #13, #14.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_reward_curve(episode_rewards: list[float], title: str = "Training Reward Curve", save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episode_rewards, alpha=0.4, label="Episode reward")
    # Rolling average
    window = max(1, len(episode_rewards) // 50)
    rolling = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    ax.plot(range(window - 1, len(episode_rewards)), rolling, label=f"Rolling avg ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Revenue")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_price_trajectory(prices: list[float], days: list[int], save_path: str | None = None) -> None:
    """Price vs days_until_departure for a sample episode — Issue #13."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(days, prices, marker="o", linewidth=1.5)
    ax.invert_xaxis()  # days count down to 0
    ax.set_xlabel("Days Until Departure")
    ax.set_ylabel("Price ($)")
    ax.set_title("Learned Price Trajectory")
    ax.axvline(x=5, color="red", linestyle="--", alpha=0.6, label="Late-deadline zone")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_revenue_distribution(revenue_dict: dict[str, np.ndarray], save_path: str | None = None) -> None:
    """Revenue distribution histogram per strategy — Issue #14."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, revenues in revenue_dict.items():
        sns.kdeplot(revenues, ax=ax, label=f"{label} (μ={revenues.mean():.0f})")
    ax.set_xlabel("Total Season Revenue ($)")
    ax.set_ylabel("Density")
    ax.set_title("Revenue Distribution by Strategy (1,000 Seasons)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
