"""Hypothesis builders for the trap-bandit experiment."""
from __future__ import annotations

import numpy as np

from ibrl.infrabayesian import AMeasure, Infradistribution
from ibrl.infrabayesian.world_models.joint_bandit_world_model import (
    JointBanditComponent,
    JointBanditWorldModel,
)


def beta_grid_weights(p_grid: np.ndarray, a: float = 2.0, b: float = 2.0) -> np.ndarray:
    raw = p_grid ** (a - 1) * (1 - p_grid) ** (b - 1)
    return raw / raw.sum()


def make_p_grid(num_grid: int = 19, p_cat: float = 0.01) -> np.ndarray:
    return np.linspace(0.05, 1.0 - p_cat - 0.05, num_grid)


def make_family_dist(
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
            components.append(JointBanditComponent(world_type, float(p1), float(p2), p_cat))
            weights.append(float(p_weights[i] * p_weights[j]))
    params = wm.make_params(components, np.asarray(weights, dtype=float))
    return Infradistribution([AMeasure(params)], world_model=wm)


def make_trap_bandit_hypotheses(
    *,
    num_grid: int = 19,
    p_cat: float = 0.01,
    p_beta: tuple[float, float] = (2.0, 2.0),
) -> tuple[JointBanditWorldModel, Infradistribution, Infradistribution]:
    wm = JointBanditWorldModel()
    p_grid = make_p_grid(num_grid, p_cat)
    p_weights = beta_grid_weights(p_grid, *p_beta)
    safe = make_family_dist(
        wm, world_type="safe", p_grid=p_grid, p_weights=p_weights, p_cat=p_cat
    )
    risky = make_family_dist(
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
