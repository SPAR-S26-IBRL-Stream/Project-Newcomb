"""Reusable hypothesis builders for infrabayesian world models."""

from .trap_bandit import (
    OUTCOME_CATASTROPHE,
    OUTCOME_ONE,
    OUTCOME_ZERO,
    beta_grid_weights,
    make_bayesian_hypothesis,
    make_ib_hypothesis,
    make_p_grid,
    make_trap_bandit_family,
    make_trap_bandit_hypotheses,
    trap_bandit_probs,
)

__all__ = [
    "OUTCOME_CATASTROPHE",
    "OUTCOME_ONE",
    "OUTCOME_ZERO",
    "beta_grid_weights",
    "make_bayesian_hypothesis",
    "make_ib_hypothesis",
    "make_p_grid",
    "make_trap_bandit_family",
    "make_trap_bandit_hypotheses",
    "trap_bandit_probs",
]
