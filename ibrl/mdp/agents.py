"""Four MDP agents: Robust DP (one-shot), Bayesian DP, Thompson DP (PSRL),
and Infrabayesian DP (online polytope shrinkage with re-planning).

The interesting comparison is RobustDPAgent (no online updating) vs
IBDPAgent — both start from the same defensible initial polytope, both
plan against worst-case successor distributions; only IB re-runs the
polytope intersection (+ replans) each episode. That's the dynamic-
consistency contribution.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import MDPAgent
from .interval_belief import IntervalKernelBelief
from .value_iteration import (
    value_iteration,
    lower_expectation_value_iteration,
    posterior_sample_value_iteration,
)


class _BaseDPAgent(MDPAgent):
    """Shared infrastructure: cache (V, policy) from plan(), use the
    deterministic greedy action in act(), default observe is no-op."""

    def __init__(self, *args, R: NDArray, terminal_mask: NDArray,
                 gamma: float = 0.95, vi_tol: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.R = np.asarray(R, dtype=np.float64)
        self.terminal_mask = np.asarray(terminal_mask, dtype=np.float64)
        self.gamma = float(gamma)
        self.vi_tol = float(vi_tol)
        self._policy: NDArray | None = None

    def act(self, state: int) -> int:
        # Greedy on cached policy; tie-break by RNG for determinism + diversity
        probs = self._policy[state]
        best = np.flatnonzero(probs == probs.max())
        return int(self.random.choice(best))

    def observe(self, state: int, action: int, next_state: int,
                reward: float, done: bool) -> None:
        pass


class RobustDPAgent(_BaseDPAgent):
    """One-shot robust DP. Plans against a fixed initial polytope; never
    updates and never re-plans. The dynamic-consistency comparison
    target."""

    def __init__(self, *args, initial_polytope: tuple[NDArray, NDArray],
                 **kwargs):
        super().__init__(*args, **kwargs)
        p_lo, p_hi = initial_polytope
        self._p_lo = np.asarray(p_lo, dtype=np.float64)
        self._p_hi = np.asarray(p_hi, dtype=np.float64)

    def plan(self) -> None:
        if self._policy is not None:
            return  # already planned, never re-plan
        _, self._policy = lower_expectation_value_iteration(
            self._p_lo, self._p_hi, self.R,
            gamma=self.gamma, terminal_mask=self.terminal_mask,
            tol=self.vi_tol)

    def reset_belief(self) -> None:
        super().reset_belief()
        self._policy = None  # full reset re-plans next call


class BayesianDPAgent(_BaseDPAgent):
    """Posterior-mean kernel + standard VI. Greedy."""

    def __init__(self, *args, belief: IntervalKernelBelief, **kwargs):
        super().__init__(*args, **kwargs)
        self.belief = belief

    def plan(self) -> None:
        P = self.belief.posterior_mean()
        _, self._policy = value_iteration(
            P, self.R, gamma=self.gamma,
            terminal_mask=self.terminal_mask, tol=self.vi_tol)

    def observe(self, state, action, next_state, reward, done):
        self.belief.update(state, action, next_state)

    def reset_belief(self) -> None:
        super().reset_belief()
        self.belief = IntervalKernelBelief(
            num_states=self.belief.num_states,
            num_actions=self.belief.num_actions,
            alpha_init=self.belief.alpha_init,
            initial_polytope=(self.belief._initial_p_lo, self.belief._initial_p_hi)
            if self.belief._initial_p_lo is not None else None,
        )


class ThompsonDPAgent(_BaseDPAgent):
    """PSRL: sample one kernel from posterior at episode start, plan greedily."""

    def __init__(self, *args, belief: IntervalKernelBelief, **kwargs):
        super().__init__(*args, **kwargs)
        self.belief = belief

    def plan(self) -> None:
        _, self._policy = posterior_sample_value_iteration(
            self.belief, self.R, rng=self.random,
            gamma=self.gamma, terminal_mask=self.terminal_mask,
            tol=self.vi_tol)

    def observe(self, state, action, next_state, reward, done):
        self.belief.update(state, action, next_state)

    def reset_belief(self) -> None:
        super().reset_belief()
        self.belief = IntervalKernelBelief(
            num_states=self.belief.num_states,
            num_actions=self.belief.num_actions,
            alpha_init=self.belief.alpha_init,
            initial_polytope=(self.belief._initial_p_lo, self.belief._initial_p_hi)
            if self.belief._initial_p_lo is not None else None,
        )


class IBDPAgent(_BaseDPAgent):
    """Lower-expectation VI on a Hoeffding-shrinking polytope intersected
    with the same initial_polytope RobustDPAgent uses. Updates and replans
    every episode — the dynamic-consistency contribution."""

    def __init__(self, *args, belief: IntervalKernelBelief,
                 confidence: float = 0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.belief = belief
        assert belief._initial_p_lo is not None, (
            "IBDPAgent requires an initial_polytope on its IntervalKernelBelief")
        self.confidence = float(confidence)

    def plan(self) -> None:
        p_lo, p_hi = self.belief.polytope(confidence=self.confidence)
        _, self._policy = lower_expectation_value_iteration(
            p_lo, p_hi, self.R, gamma=self.gamma,
            terminal_mask=self.terminal_mask, tol=self.vi_tol)

    def observe(self, state, action, next_state, reward, done):
        self.belief.update(state, action, next_state)

    def reset_belief(self) -> None:
        super().reset_belief()
        self.belief = IntervalKernelBelief(
            num_states=self.belief.num_states,
            num_actions=self.belief.num_actions,
            alpha_init=self.belief.alpha_init,
            initial_polytope=(self.belief._initial_p_lo, self.belief._initial_p_hi),
        )
