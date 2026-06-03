"""
Unit tests for DynamicPricingEnv — Issue #4.
"""
import numpy as np
import pytest

from env import DynamicPricingEnv


@pytest.fixture
def env():
    e = DynamicPricingEnv(max_inventory=10, max_days=5, price_levels=[50, 100, 150])
    yield e
    e.close()


def test_reset_returns_full_state(env):
    obs, info = env.reset()
    assert obs[0] == 10  # inventory
    assert obs[1] == 5   # days


def test_step_decrements_days(env):
    env.reset()
    obs, _, _, _, _ = env.step(0)
    assert obs[1] == 4


def test_inventory_never_negative(env):
    env.reset()
    for _ in range(100):
        obs, _, done, _, _ = env.step(0)
        assert obs[0] >= 0
        if done:
            break


def test_episode_terminates_at_day_zero(env):
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(1)
        done = terminated or truncated
        steps += 1
    assert steps <= 10  # max_days=5, inventory=10


def test_reward_is_non_negative(env):
    env.reset()
    for _ in range(5):
        _, reward, done, _, _ = env.step(2)
        assert reward >= 0
        if done:
            break


def test_observation_space_bounds(env):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)


def test_smoke_100_episodes():
    """100-episode random policy smoke test."""
    e = DynamicPricingEnv()
    for _ in range(100):
        obs, _ = e.reset()
        done = False
        while not done:
            action = e.action_space.sample()
            obs, _, terminated, truncated, _ = e.step(action)
            done = terminated or truncated
    e.close()
