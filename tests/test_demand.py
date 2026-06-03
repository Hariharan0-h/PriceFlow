"""
Validate demand function — Issue #2 acceptance criteria.
"""
import numpy as np
from env.demand import demand_probability


def test_monotonic_price_decrease():
    """Higher price → lower (or equal) demand probability."""
    prices = [50, 75, 100, 125, 150, 175, 200]
    rng = np.random.default_rng(42)
    probs = [demand_probability(p, days_remaining=15, rng=rng) for p in prices]
    # Without noise the function is strictly decreasing; with noise allow small tolerance
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1] - 0.15, (
            f"Non-monotonic: P({prices[i]})={probs[i]:.3f} < P({prices[i+1]})={probs[i+1]:.3f}"
        )


def test_output_in_unit_interval():
    for price in [50, 100, 200]:
        for days in [0, 15, 30]:
            p = demand_probability(price, days)
            assert 0.0 <= p <= 1.0
