"""Compatibility imports for promoted trap-bandit builders."""

from ibrl.infrabayesian.builders.trap_bandit import (
    beta_grid_weights,
    make_bayesian_hypothesis,
    make_ib_hypothesis,
    make_p_grid,
    make_trap_bandit_family,
    make_trap_bandit_hypotheses,
    trap_bandit_probs,
)

__all__ = [
    "beta_grid_weights",
    "make_bayesian_hypothesis",
    "make_ib_hypothesis",
    "make_p_grid",
    "make_trap_bandit_family",
    "make_trap_bandit_hypotheses",
    "trap_bandit_probs",
]
