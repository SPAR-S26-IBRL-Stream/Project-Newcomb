"""Builders for trap-bandit hypotheses."""
from __future__ import annotations

import numpy as np

from ..a_measure import AMeasure
from ..infradistribution import Infradistribution
from ..world_models.joint_bandit_world_model import (
    JointBanditComponent,
    JointBanditWorldModel,
)


OUTCOME_ZERO = 0
OUTCOME_ONE = 1
OUTCOME_CATASTROPHE = 2


def beta_grid_weights(p_grid: np.ndarray, a: float = 2.0, b: float = 2.0) -> np.ndarray:
    raw = p_grid ** (a - 1) * (1 - p_grid) ** (b - 1)
    return raw / raw.sum()


def make_p_grid(num_grid: int = 19, p_cat: float = 0.01) -> np.ndarray:
    return np.linspace(0.05, 1.0 - p_cat - 0.05, num_grid)


def trap_bandit_probs(
    p_values: np.ndarray,
    *,
    world_type: str,
    p_cat: float,
) -> np.ndarray:
    """Create per-arm outcome probabilities for one trap-bandit component."""
    assert world_type in {"safe", "risky"}
    p_values = np.asarray(p_values, dtype=float)
    assert p_values.ndim == 1
    assert np.all(p_values >= 0)
    assert np.all(p_values + p_cat <= 1)

    probs = np.zeros((len(p_values), 3))
    trapped_arm = int(np.argmax(p_values))
    for action, p in enumerate(p_values):
        if world_type == "risky" and action == trapped_arm:
            probs[action] = [1.0 - p - p_cat, p, p_cat]
        else:
            probs[action] = [1.0 - p, p, 0.0]
    return probs


def make_trap_bandit_family(
    wm: JointBanditWorldModel,
    *,
    world_type: str,
    p_grid: np.ndarray,
    p_weights: np.ndarray,
    p_cat: float,
) -> Infradistribution:
    components = []
    weights = []
    for i, p1 in enumerate(p_grid):
        for j, p2 in enumerate(p_grid):
            p_values = np.array([p1, p2], dtype=float)
            components.append(
                JointBanditComponent(
                    trap_bandit_probs(p_values, world_type=world_type, p_cat=p_cat),
                    metadata={
                        "world_type": world_type,
                        "p_values": p_values,
                        "p_cat": p_cat,
                        "trapped_arm": int(np.argmax(p_values)),
                    },
                )
            )
            weights.append(float(p_weights[i] * p_weights[j]))
    params = wm.make_params(components, np.asarray(weights, dtype=float))
    return Infradistribution([AMeasure(params)], world_model=wm)


def make_trap_bandit_hypotheses(
    *,
    num_grid: int = 19,
    p_cat: float = 0.01,
    p_beta: tuple[float, float] = (2.0, 2.0),
) -> tuple[JointBanditWorldModel, Infradistribution, Infradistribution]:
    wm = JointBanditWorldModel(num_arms=2, num_outcomes=3)
    p_grid = make_p_grid(num_grid, p_cat)
    p_weights = beta_grid_weights(p_grid, *p_beta)
    safe = make_trap_bandit_family(
        wm, world_type="safe", p_grid=p_grid, p_weights=p_weights, p_cat=p_cat
    )
    risky = make_trap_bandit_family(
        wm, world_type="risky", p_grid=p_grid, p_weights=p_weights, p_cat=p_cat
    )
    return wm, safe, risky


def make_bayesian_hypothesis(
    safe: Infradistribution,
    risky: Infradistribution,
    *,
    alpha_beta: tuple[float, float] = (2.0, 2.0),
) -> Infradistribution:
    a, b = alpha_beta
    p_risky = a / (a + b)
    return Infradistribution.mix([safe, risky], np.array([1.0 - p_risky, p_risky]))


def make_ib_hypothesis(safe: Infradistribution, risky: Infradistribution) -> Infradistribution:
    return Infradistribution.mixKU([safe, risky])
