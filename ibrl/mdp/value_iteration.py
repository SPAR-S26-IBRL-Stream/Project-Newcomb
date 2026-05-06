"""Value iteration solvers — standard, lower-expectation (robust), and
posterior-sample variants. Reward is parameterised by destination state
(immediate reward when arriving at s'), which suits the gridworld setup
where reward and trap rewards are state-attached.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _bellman_backup(P: NDArray, R: NDArray, V: NDArray, gamma: float,
                    terminal_mask: NDArray) -> NDArray:
    """One Bellman backup against fixed kernel P (S, A, S).

    Q(s, a) = sum_{s'} P[s,a,s'] * (R[s'] + gamma * V[s'] * (1 - terminal[s']))
    """
    next_value = R + gamma * V * (1.0 - terminal_mask)
    return np.einsum("sat,t->sa", P, next_value)


def value_iteration(P: NDArray, R: NDArray, *,
                    gamma: float = 0.95,
                    terminal_mask: NDArray | None = None,
                    tol: float = 1e-6,
                    max_iter: int = 2000) -> tuple[NDArray, NDArray]:
    """Standard VI on a fixed kernel.

    Arguments:
        P:              (num_states, num_actions, num_states) kernel
        R:              (num_states,) per-arrival reward
        gamma:          discount
        terminal_mask:  (num_states,) bool/0-1 — 1 means terminal (no continuation)
        tol, max_iter:  convergence parameters

    Returns:
        V:      (num_states,)
        policy: (num_states, num_actions) — one-hot greedy
    """
    P = np.asarray(P, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    num_states, num_actions, _ = P.shape
    if terminal_mask is None:
        terminal_mask = np.zeros(num_states)
    terminal_mask = np.asarray(terminal_mask, dtype=np.float64)

    V = np.zeros(num_states)
    for _ in range(max_iter):
        Q = _bellman_backup(P, R, V, gamma, terminal_mask)
        V_new = Q.max(axis=1)
        # Pin terminals to 0 (no continuation)
        V_new = V_new * (1.0 - terminal_mask)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = _bellman_backup(P, R, V, gamma, terminal_mask)
    policy = np.zeros((num_states, num_actions))
    best = Q.argmax(axis=1)
    policy[np.arange(num_states), best] = 1.0
    return V, policy


def lower_expectation_kernel(p_lower: NDArray, p_upper: NDArray,
                             value_for_sort: NDArray) -> NDArray:
    """Per (s, a), find the worst-case successor distribution
    p ∈ [p_lower, p_upper] with sum(p) = 1 that minimises
    Σ p[s'] · value_for_sort[s'].

    Closed-form (LP at a vertex of the box-simplex polytope): assign
    p_upper to the lowest-value successors and p_lower to the highest,
    with one fractional successor that closes the sum-to-1 constraint.

    `value_for_sort` is the **next-state contribution** the Bellman
    backup will weight — i.e. R[s'] + gamma · V[s'] · (1 − terminal[s']).
    Pass that, not V itself; otherwise terminal states (V pinned to 0)
    rank as "low value" and the LP misroutes probability into them
    even when their reward is large positive (e.g. the reward terminal).

    Arguments:
        p_lower, p_upper:  (num_states, num_actions, num_states) each
        value_for_sort:    (num_states,)

    Returns:
        p_worst:           (num_states, num_actions, num_states) — the
                           worst-case kernel for *this* next-value vector.

    Raises ValueError if the box [p_lower, p_upper] is infeasible
    (sum_lower > 1 or sum_upper < 1) for some (s, a).
    """
    p_lower = np.asarray(p_lower, dtype=np.float64)
    p_upper = np.asarray(p_upper, dtype=np.float64)
    value_for_sort = np.asarray(value_for_sort, dtype=np.float64)
    S, A, S2 = p_lower.shape
    assert S == S2 and p_upper.shape == p_lower.shape

    sum_lower = p_lower.sum(axis=2)
    sum_upper = p_upper.sum(axis=2)
    if np.any(sum_lower > 1.0 + 1e-9):
        raise ValueError(f"infeasible polytope: sum(p_lower) > 1 somewhere "
                         f"(max={sum_lower.max():.6f})")
    if np.any(sum_upper < 1.0 - 1e-9):
        raise ValueError(f"infeasible polytope: sum(p_upper) < 1 somewhere "
                         f"(min={sum_upper.min():.6f})")

    order = np.argsort(value_for_sort)
    p_worst = np.zeros_like(p_lower)
    for s in range(S):
        for a in range(A):
            lo = p_lower[s, a]
            hi = p_upper[s, a]
            assignment = lo.copy()
            remaining = 1.0 - lo.sum()
            for k in order:
                if remaining <= 1e-12:
                    break
                slack = hi[k] - lo[k]
                take = min(slack, remaining)
                assignment[k] += take
                remaining -= take
            p_worst[s, a] = assignment
    return p_worst


def lower_expectation_value_iteration(
        p_lower: NDArray, p_upper: NDArray, R: NDArray, *,
        gamma: float = 0.95,
        terminal_mask: NDArray | None = None,
        tol: float = 1e-6,
        max_iter: int = 2000) -> tuple[NDArray, NDArray]:
    """Robust VI: at each Bellman backup, the adversary picks the worst-case
    successor distribution from [p_lower, p_upper] for the current V.

    Returns (V, policy) — V is the lower-expectation optimal value,
    policy is the maximin greedy policy.
    """
    p_lower = np.asarray(p_lower, dtype=np.float64)
    p_upper = np.asarray(p_upper, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    S = R.shape[0]
    if terminal_mask is None:
        terminal_mask = np.zeros(S)
    terminal_mask = np.asarray(terminal_mask, dtype=np.float64)

    V = np.zeros(S)
    for _ in range(max_iter):
        next_value = R + gamma * V * (1.0 - terminal_mask)
        P_worst = lower_expectation_kernel(p_lower, p_upper, next_value)
        Q = _bellman_backup(P_worst, R, V, gamma, terminal_mask)
        V_new = Q.max(axis=1)
        V_new = V_new * (1.0 - terminal_mask)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    next_value = R + gamma * V * (1.0 - terminal_mask)
    P_worst = lower_expectation_kernel(p_lower, p_upper, next_value)
    Q = _bellman_backup(P_worst, R, V, gamma, terminal_mask)
    policy = np.zeros((S, p_lower.shape[1]))
    best = Q.argmax(axis=1)
    policy[np.arange(S), best] = 1.0
    return V, policy


def posterior_sample_value_iteration(
        belief, R: NDArray, *,
        rng: np.random.Generator,
        gamma: float = 0.95,
        terminal_mask: NDArray | None = None,
        tol: float = 1e-6) -> tuple[NDArray, NDArray]:
    """Thompson / PSRL: sample one kernel from belief.posterior_sample, then VI."""
    P = belief.posterior_sample_full(rng)
    return value_iteration(P, R, gamma=gamma, terminal_mask=terminal_mask, tol=tol)
