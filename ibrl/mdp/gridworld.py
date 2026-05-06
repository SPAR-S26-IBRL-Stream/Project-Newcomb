"""Robust gridworld with interval-uncertain transitions and an adversary
that picks the worst-case kernel from inside the interval.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import MDPEnvironment
from .value_iteration import value_iteration, lower_expectation_kernel


# Action indices: 0=N, 1=E, 2=S, 3=W
_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _make_intended_kernel(rows: int, cols: int, p_intended: NDArray,
                          terminal_states: list[int]) -> NDArray:
    """Build the (S, A, S) kernel given a per-state-per-action intended-prob
    array `p_intended (S, A)`. Each non-intended direction (the two
    perpendiculars) gets (1 - p) / 2; walls bounce to current state.
    Terminal states absorb (P[s, a, s] = 1)."""
    S = rows * cols
    A = 4
    P = np.zeros((S, A, S))
    for s in range(S):
        if s in terminal_states:
            P[s, :, s] = 1.0
            continue
        r, c = divmod(s, cols)
        for a in range(A):
            p = p_intended[s, a]
            # Intended direction
            dr, dc = _DELTAS[a]
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < rows and 0 <= c2 < cols:
                P[s, a, r2 * cols + c2] += p
            else:
                P[s, a, s] += p
            # Two perpendiculars get (1 - p) / 2 each
            for perp_a in [(a + 1) % 4, (a - 1) % 4]:
                dr_p, dc_p = _DELTAS[perp_a]
                r3, c3 = r + dr_p, c + dc_p
                if 0 <= r3 < rows and 0 <= c3 < cols:
                    P[s, a, r3 * cols + c3] += (1.0 - p) / 2.0
                else:
                    P[s, a, s] += (1.0 - p) / 2.0
    return P


class GridworldEnvironment(MDPEnvironment):
    """Episodic 5×5-by-default gridworld with a reward state, a trap, and
    interval-uncertain transitions.

    Transition model: at every (s, a), the *intended* successor receives
    probability `p_eff(s, a)`, each perpendicular gets `(1 − p_eff)/2`,
    walls bounce. The agent's polytope says
    `p_eff(s, a) ∈ [p_nominal − epsilon, p_nominal + epsilon]`. The
    adversary picks the actual `p_eff(s, a)` from that interval.

    Adversary modes:
        "static":             pick worst-case once at construction against
                              the env's own oracle V*. Stationary kernel
                              across episodes — Bayesian / Thompson can
                              learn it in the limit.
        "per_episode_visit":  re-pick at each `reset_episode()` against
                              V* shaped by the agent's previous-episode
                              visit distribution. Non-stationary kernel —
                              past observations no longer fully predict
                              the next episode's transitions, which is the
                              regime where IB's worst-case-with-updating
                              should differentiate.
    """

    def __init__(self, *,
                 rows: int = 5,
                 cols: int = 5,
                 reward_pos: tuple[int, int] = (4, 4),
                 trap_pos: tuple[int, int] = (2, 2),
                 start_pos: tuple[int, int] = (0, 0),
                 p_nominal: float = 0.8,
                 epsilon: float = 0.05,
                 step_cost: float = -0.01,
                 reward_value: float = 1.0,
                 trap_value: float = -1.0,
                 adversarial: bool = True,
                 adversary_mode: str = "static",
                 gamma: float = 0.95,
                 seed: int = 0x89ABCDEF,
                 verbose: int = 0):
        super().__init__(num_states=rows * cols, num_actions=4,
                         seed=seed, verbose=verbose)
        assert 0.0 < p_nominal <= 1.0
        assert 0.0 <= epsilon <= min(p_nominal, 1.0 - p_nominal) + 1e-9
        assert adversary_mode in {"static", "per_episode_visit"}
        self.rows = rows
        self.cols = cols
        self.reward_pos = reward_pos
        self.trap_pos = trap_pos
        self.start_pos = start_pos
        self.p_nominal = float(p_nominal)
        self.epsilon = float(epsilon)
        self.step_cost = float(step_cost)
        self.reward_value = float(reward_value)
        self.trap_value = float(trap_value)
        self.adversarial = bool(adversarial)
        self.adversary_mode = adversary_mode
        self.gamma = float(gamma)

        self.reward_idx = reward_pos[0] * cols + reward_pos[1]
        self.trap_idx = trap_pos[0] * cols + trap_pos[1]
        self.start_idx = start_pos[0] * cols + start_pos[1]
        self.terminal_states = [self.reward_idx, self.trap_idx]
        self.terminal_mask = np.zeros(self.num_states)
        for t in self.terminal_states:
            self.terminal_mask[t] = 1.0

        # Per-arrival reward vector
        self.R = np.full(self.num_states, self.step_cost)
        self.R[self.reward_idx] = self.reward_value
        self.R[self.trap_idx] = self.trap_value

        # Visit-distribution bookkeeping for the per_episode_visit adversary;
        # initialised here so reset_episode() can read them on episode 0.
        self._prev_visit_counts = np.zeros(self.num_states)
        self._curr_visit_counts = np.zeros(self.num_states)
        self._current_state = self.start_idx

    # ── Bookkeeping ────────────────────────────────────────────────────

    @property
    def polytope_bounds(self) -> tuple[NDArray, NDArray]:
        """[p_lower, p_upper] for the *intended-direction probability* per
        (s, a). Shape (num_states, num_actions). The full kernel polytope
        is derived from this via _make_intended_kernel."""
        p_lo = np.full((self.num_states, self.num_actions),
                       self.p_nominal - self.epsilon)
        p_hi = np.full((self.num_states, self.num_actions),
                       self.p_nominal + self.epsilon)
        return p_lo, p_hi

    def kernel_polytope(self) -> tuple[NDArray, NDArray]:
        """Full per-(s,a,s') polytope [P_lo, P_hi] derived from intended-prob
        bounds. This is the polytope the agent's IBDPAgent and RobustDPAgent
        plan against; it is also the correct prior interval to give the
        agent's IntervalKernelBelief."""
        p_lo, p_hi = self.polytope_bounds
        P_at_lo = _make_intended_kernel(self.rows, self.cols, p_lo, self.terminal_states)
        P_at_hi = _make_intended_kernel(self.rows, self.cols, p_hi, self.terminal_states)
        # When intended probability ↑, perpendiculars ↓ — element-wise min/max
        # over the two endpoints gives the per-cell polytope bounds.
        P_min = np.minimum(P_at_lo, P_at_hi)
        P_max = np.maximum(P_at_lo, P_at_hi)
        return P_min, P_max

    # ── Adversary ──────────────────────────────────────────────────────

    def _make_oracle_kernel(self, V_for_adversary: NDArray) -> NDArray:
        """Pick the worst-case intended-probability per (s, a) against the
        given V, then assemble the kernel."""
        if not self.adversarial:
            P_int = np.full((self.num_states, self.num_actions), self.p_nominal)
            return _make_intended_kernel(self.rows, self.cols, P_int, self.terminal_states)

        # For each (s, a), compute the V-weighted next-state expectation at
        # both p_lo and p_hi; pick the one that minimises agent value.
        p_lo, p_hi = self.polytope_bounds
        P_lo = _make_intended_kernel(self.rows, self.cols, p_lo, self.terminal_states)
        P_hi = _make_intended_kernel(self.rows, self.cols, p_hi, self.terminal_states)
        next_value = self.R + self.gamma * V_for_adversary * (1.0 - self.terminal_mask)
        EV_lo = np.einsum("sat,t->sa", P_lo, next_value)
        EV_hi = np.einsum("sat,t->sa", P_hi, next_value)
        choose_hi = EV_hi < EV_lo
        P_int = np.where(choose_hi, p_hi, p_lo)
        return _make_intended_kernel(self.rows, self.cols, P_int, self.terminal_states)

    def _solve_oracle_value(self) -> NDArray:
        """Self-referencing fixed-point: V* under the worst-case kernel
        chosen against V* itself. We compute this via lower-expectation VI
        on the (P_min, P_max) cell-polytope, which is equivalent."""
        P_min, P_max = self.kernel_polytope()
        V, _ = value_iteration(self.true_kernel, self.R,
                               gamma=self.gamma,
                               terminal_mask=self.terminal_mask)
        return V

    def _refresh_kernel(self, V_for_adversary: NDArray | None = None,
                        max_outer_iters: int = 50,
                        tol: float = 1e-6) -> None:
        """Recompute self.true_kernel (the actual transition kernel that
        env.step samples from).

        Static-adversary mode: iterate to the fixed point
            kernel ← worst-case kernel against V
            V ← V*(kernel)
        until the kernel stops changing — this is the robust DP fixed point
        the static adversary commits to. Without this, the kernel chosen
        from a non-fixed-point V is "overcommitted pessimism" against an
        outdated V; convergence makes the env's V coincide with the
        polytope's robust V*, which is what the IB agent expects to plan
        against.

        Per-episode-visit mode: a single pass against V_for_adversary is
        appropriate (the adversary commits non-stationarily each episode).
        """
        if V_for_adversary is not None:
            self.true_kernel = self._make_oracle_kernel(V_for_adversary)
        else:
            # Bootstrap with nominal kernel, then iterate kernel/V to fixed point
            P_curr = _make_intended_kernel(
                self.rows, self.cols,
                np.full((self.num_states, self.num_actions), self.p_nominal),
                self.terminal_states)
            for _ in range(max_outer_iters):
                V_curr, _ = value_iteration(P_curr, self.R,
                                            gamma=self.gamma,
                                            terminal_mask=self.terminal_mask)
                P_next = self._make_oracle_kernel(V_curr)
                if np.max(np.abs(P_next - P_curr)) < tol:
                    P_curr = P_next
                    break
                P_curr = P_next
            self.true_kernel = P_curr
        # Oracle V cache is stale whenever the kernel changes
        if hasattr(self, "_oracle_V_cache"):
            del self._oracle_V_cache

    # ── MDPEnvironment overrides ───────────────────────────────────────

    def reset(self) -> int:
        # Reset adversary state and visit counts BEFORE super().reset(),
        # because super().reset() triggers reset_episode() which reads them.
        self._prev_visit_counts = np.zeros(self.num_states)
        self._curr_visit_counts = np.zeros(self.num_states)
        self._refresh_kernel(V_for_adversary=None)
        return super().reset()

    def reset_episode(self) -> int:
        # super().reset_episode() bumps seed and re-inits RNG.
        # Use parent class directly to skip the abstract _initial_state at
        # MDPEnvironment.reset_episode (we override return below).
        self.seed += 1
        self.random = np.random.default_rng(seed=self.seed)

        if self.adversary_mode == "per_episode_visit":
            # Repick worst-case kernel against V* shaped by the agent's
            # previous-episode visit distribution.
            if self._prev_visit_counts.sum() > 0:
                visit_dist = self._prev_visit_counts / self._prev_visit_counts.sum()
                V_curr, _ = value_iteration(self.true_kernel, self.R,
                                            gamma=self.gamma,
                                            terminal_mask=self.terminal_mask)
                V_shaped = V_curr * (1.0 + visit_dist)
                self._refresh_kernel(V_for_adversary=V_shaped)

        # Roll over visit counts for the new episode
        self._prev_visit_counts = self._curr_visit_counts.copy()
        self._curr_visit_counts = np.zeros(self.num_states)
        return self._initial_state()

    def _initial_state(self) -> int:
        self._current_state = self.start_idx
        return self.start_idx

    def step(self, action: int) -> tuple[int, float, bool]:
        assert 0 <= action < self.num_actions
        # current state from caller's perspective tracked by the simulator;
        # we infer "current state" from RNG draw against the cached kernel.
        # Convention: the simulator passes state via self._current_state.
        s = self._current_state
        probs = self.true_kernel[s, action]
        next_state = int(self.random.choice(self.num_states, p=probs))
        self._current_state = next_state
        reward = float(self.R[next_state])
        done = bool(self.terminal_mask[next_state])
        self._curr_visit_counts[next_state] += 1
        return next_state, reward, done

    def get_oracle_value(self, gamma: float | None = None) -> NDArray:
        """V* under the env's actual transition kernel."""
        if gamma is not None and abs(gamma - self.gamma) > 1e-12:
            # Caller asked for a different gamma — recompute on demand.
            V, _ = value_iteration(self.true_kernel, self.R,
                                   gamma=gamma, terminal_mask=self.terminal_mask)
            return V
        if not hasattr(self, "_oracle_V_cache"):
            V, _ = value_iteration(self.true_kernel, self.R,
                                   gamma=self.gamma,
                                   terminal_mask=self.terminal_mask)
            self._oracle_V_cache = V
        return self._oracle_V_cache

