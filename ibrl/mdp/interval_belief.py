"""Dirichlet posterior over transition kernels with optional initial polytope."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as scipy_beta


class IntervalKernelBelief:
    """Per (s, a) Dirichlet posterior over successor distribution.

    Three accessors:
        posterior_mean(s, a)       — Dirichlet mean
        posterior_sample_full(rng) — sample full kernel (S, A, S) for PSRL
        polytope(confidence=0.95)  — per-(s,a,s') Hoeffding interval
                                      [p_low, p_high], intersected with the
                                      constructor's `initial_polytope` if
                                      provided. Returns (p_lower, p_upper)
                                      both shape (S, A, S).

    `alpha_init=0.5` (Jeffreys) is the default rather than 1.0 to keep the
    polytope from blowing up to the full simplex at unvisited (s, a).
    """

    def __init__(self, num_states: int, num_actions: int, *,
                 alpha_init: float = 0.5,
                 initial_polytope: tuple[NDArray, NDArray] | None = None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha_init = float(alpha_init)
        # Per (s, a), a Dirichlet over successor states. Initialise with
        # alpha_init only at cells the initial polytope deems possible
        # (initial_p_hi > 0); structurally-impossible successors get α=0
        # so the posterior mean isn't depressed by their prior mass.
        self.alpha = np.full((num_states, num_actions, num_states),
                             self.alpha_init, dtype=np.float64)
        if initial_polytope is not None:
            p_lo, p_hi = initial_polytope
            p_lo = np.asarray(p_lo, dtype=np.float64)
            p_hi = np.asarray(p_hi, dtype=np.float64)
            assert p_lo.shape == (num_states, num_actions, num_states)
            assert p_hi.shape == (num_states, num_actions, num_states)
            self._initial_p_lo = p_lo.copy()
            self._initial_p_hi = p_hi.copy()
            # Mask out structurally-impossible successors (initial_p_hi = 0)
            self.alpha = np.where(p_hi > 0, self.alpha, 0.0)
        else:
            self._initial_p_lo = None
            self._initial_p_hi = None

    def update(self, state: int, action: int, next_state: int) -> None:
        self.alpha[state, action, next_state] += 1.0

    def posterior_mean(self) -> NDArray:
        """Full posterior-mean kernel, shape (S, A, S)."""
        # Avoid divide-by-zero at (s, a) pairs where every cell has α=0
        # (e.g. terminal state with no observations yet); fall back to a
        # uniform-on-self distribution as a safe default.
        sums = self.alpha.sum(axis=2, keepdims=True)
        sums = np.where(sums > 0, sums, 1.0)
        mean = self.alpha / sums
        return mean

    def posterior_sample_full(self, rng: np.random.Generator) -> NDArray:
        """Sample one full kernel (S, A, S) by drawing each (s,a) row from
        its Dirichlet posterior. Cells with α=0 are excluded — they
        contribute 0 probability."""
        P = np.zeros_like(self.alpha)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                row = self.alpha[s, a]
                support = row > 0
                if support.any():
                    P[s, a, support] = rng.dirichlet(row[support])
                else:
                    # Degenerate; safe default = stay at s (e.g. terminal)
                    P[s, a, s] = 1.0
        return P

    def polytope(self, confidence: float = 0.95) -> tuple[NDArray, NDArray]:
        """Hoeffding interval on each successor probability, intersected
        with the initial polytope if one was provided.

        For a Dirichlet posterior with concentration parameter alpha[s,a],
        we use n = sum(alpha[s,a]) - num_states*alpha_init as the effective
        observation count and apply a Hoeffding-style radius
        ``sqrt(log(2/(1-confidence)) / (2 max(n, 1)))``.

        At unvisited (s, a) (n ≈ 0), the radius is large and the
        intersection with the initial polytope keeps the polytope bounded.
        """
        assert 0.0 < confidence < 1.0

        # Per-cell Beta credible interval. For a Dirichlet(α) posterior the
        # marginal of the k-th component is Beta(α[k], Σα − α[k]). The
        # (1-confidence)/2 and (1+confidence)/2 quantiles of that Beta give
        # the per-cell credible interval. Tighter than Hoeffding by orders
        # of magnitude when α is concentrated, which is what makes the
        # IB polytope visibly shrink across episodes.
        alpha_k = self.alpha
        alpha_sum = alpha_k.sum(axis=2, keepdims=True)
        beta_k = np.maximum(alpha_sum - alpha_k, 1e-12)
        # Cells with α=0 (structurally impossible): credible interval = [0, 0]
        active = alpha_k > 0
        tail = (1.0 - confidence) / 2.0
        # scipy_beta.ppf handles the broadcasted shape (S, A, S) directly
        p_lo = np.where(active, scipy_beta.ppf(tail, alpha_k, beta_k), 0.0)
        p_hi = np.where(active, scipy_beta.ppf(1.0 - tail, alpha_k, beta_k), 0.0)
        # ppf can return NaN at α≈0; guard
        p_lo = np.nan_to_num(p_lo, nan=0.0)
        p_hi = np.nan_to_num(p_hi, nan=0.0)

        if self._initial_p_lo is not None:
            cand_lo = np.maximum(p_lo, self._initial_p_lo)
            cand_hi = np.minimum(p_hi, self._initial_p_hi)
            # Ensure cell-wise p_lo <= p_hi
            cand_lo = np.minimum(cand_lo, cand_hi)
            # If the per-cell intersection makes the joint polytope at some
            # (s, a) infeasible (sum_hi < 1 or sum_lo > 1), the data is too
            # sparse to override the prior — fall back to the initial
            # polytope at that (s, a). Per-(s, a) check so heavily-visited
            # rows still benefit from the Beta tightening.
            sum_lo = cand_lo.sum(axis=2)
            sum_hi = cand_hi.sum(axis=2)
            infeasible = (sum_hi < 1.0) | (sum_lo > 1.0)
            mask = infeasible[..., None]  # broadcast over successor axis
            p_lo = np.where(mask, self._initial_p_lo, cand_lo)
            p_hi = np.where(mask, self._initial_p_hi, cand_hi)

        return p_lo, p_hi
