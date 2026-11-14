"""Group advantage functions for the GRPO arms.

The cookbook's `compute_advantages` centres rewards within a group and leaves
them unscaled (tinker-cookbook 0.6.1, `rl/data_processing.py`):

    A_i = r_i - mean(r_1 ... r_G)

Arm a2 divides by the group standard deviation with a floor, per the GRPO
proposal in the README:

    A_i = ( r_i - mean(r) ) / max( std(r), sigma_min )

The floor stops noise in an almost-constant group from becoming a large
gradient. Pure math here so the offline tests need no torch; `train_rl.py`
wraps this into the cookbook's expected signature (one tensor per group).
"""
from __future__ import annotations

import math


def group_advantages(rewards: list[float], sigma_min: float = 0.25) -> list[float]:
    """Centred, std-scaled advantages for one group of rewards."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    denom = max(math.sqrt(var), sigma_min)
    return [(r - mean) / denom for r in rewards]
