"""SupraPOMDPWorldModel — world model for a finite credal mixture of point-valued POMDPs."""
from __future__ import annotations

import numpy as np

from .base import WorldModel
from ...outcome import Outcome


class SupraPOMDPWorldModel(WorldModel):
    """
    World model for a stateful POMDP with optional policy-dependent kernels.

    Belief state: list of (belief_vec, log_marginal) per hypothesis component,
        or None before the first update.

    Hypothesis params — single hypothesis: tuple (T, B, theta_0, R) where
        T:       shape (|S|, |A|, |S|) or callable policy → that shape
        B:       shape (|S|, |O|)      or callable policy → that shape
        theta_0: shape (|S|,)          or callable policy → that shape
        R:       shape (|S|, |A|, |S|) — always static

    Mixed params (after Infradistribution.mix): list of (params, prior_weight).

    Construct params via make_params(T, B, theta_0, R).

    Known approximation: compute_q_values uses state-MDP value iteration
    projected via V(b) = b @ V_s, which is exact for fully-observable POMDPs
    and an upper-bound approximation for partially-observable ones.

    Out of scope: set-valued (infrakernel) transitions; per-step adversarial
    transitions (Appel-Kosoy halfspace RMDPs).
    """

    def __init__(self, num_states: int, num_actions: int, num_obs: int,
                 discount: float = 0.95, value_iter_tol: float = 1e-6,
                 value_iter_max: int = 1000):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.discount = discount
        self.value_iter_tol = value_iter_tol
        self.value_iter_max = value_iter_max
        # Cache: (id(component_params), policy_bytes) → V vector.
        self._v_cache: dict = {}

    @staticmethod
    def _resolve(kernel_or_fn, policy):
        """Evaluate kernel_or_fn at policy if callable, else return as-is."""
        return kernel_or_fn(policy) if callable(kernel_or_fn) else kernel_or_fn

    # ── Param construction ────────────────────────────────────────────────────

    def make_params(self, T, B, theta_0, R: np.ndarray):
        """Validate shapes and stochasticity, then bundle into a tuple.

        T, B, and theta_0 may each be an np.ndarray of the canonical shape
        or a callable policy → np.ndarray of that shape. Callables are
        sample-validated on the uniform policy. R must be a static array.
        """
        raise NotImplementedError

    def mix_params(self, params_list: list, coefficients: np.ndarray):
        """Mix point-valued POMDP hypotheses into a credal mixture.

        Returns list of (params, prior_weight) tuples. Flattens nested mixtures.
        """
        raise NotImplementedError

    def event_index(self, outcome: Outcome, action: int) -> int:
        """Map an outcome to its discrete observation index."""
        raise NotImplementedError

    # ── Belief state ──────────────────────────────────────────────────────────

    def initial_state(self):
        """None — lazily initialised on first update_state call."""
        raise NotImplementedError

    def is_initial(self, state) -> bool:
        raise NotImplementedError

    def _initial_belief(self, component_params, policy=None) -> np.ndarray:
        """Initial belief for one component, resolving Θ₀ against policy."""
        raise NotImplementedError

    # ── Bayesian filter ───────────────────────────────────────────────────────

    def update_state(self, state, outcome: Outcome, action: int,
                     policy: np.ndarray, params=None):
        """Standard POMDP Bayesian filter applied per component.

        State shape: list[(belief_vec, log_marginal)], one per component.
        """
        raise NotImplementedError

    def _posterior_weights(self, state, params) -> np.ndarray:
        """Posterior credal weights: weights[k] ∝ prior_weight[k] · P(history|k)."""
        raise NotImplementedError

    # ── IB likelihood (one-step, used by Infradistribution.update) ───────────

    def compute_likelihood(self, belief_state, outcome: Outcome, params,
                           action: int, policy=None) -> float:
        """P(observation | belief, action) averaged over components by posterior weights."""
        raise NotImplementedError

    def compute_expected_reward(self, belief_state, reward_function: np.ndarray,
                                params, action: int, policy=None) -> float:
        """E[reward_function[next_obs] | belief, action] — one-step expectation.

        Used by Infradistribution.update for IB normalisation. Does NOT
        run value iteration. See compute_q_values for multi-step planning.
        """
        raise NotImplementedError

    # ── Planning (multi-step, used by SupraPOMDPAgent._expected_rewards) ─────

    def compute_q_values(self, belief_state, params,
                         policy: np.ndarray | None = None) -> np.ndarray:
        """Multi-step Q-values Q(b, a) for each action under policy π.

        Runs value iteration per component, weighted by posterior credal weights.
        The candidate policy π is threaded into policy-dependent kernels so
        that Newcomb-family predictors see the correct committed policy.

        Returns np.ndarray of shape (num_actions,).
        """
        raise NotImplementedError

    def _value_iteration(self, T: np.ndarray, R: np.ndarray,
                         policy: np.ndarray, cache_key=None) -> np.ndarray:
        """Standard infinite-horizon value iteration under fixed stochastic policy π.

        V[s] = Σ_a π[a] · Σ_{s'} T[s,a,s'] · (R[s,a,s'] + γ·V[s'])

        Returns shape (|S|,). Uses _v_cache keyed by (id(params), policy_bytes).
        """
        raise NotImplementedError
