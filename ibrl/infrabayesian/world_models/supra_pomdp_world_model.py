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

    def __init__(self, num_states: int, num_actions: int, num_obs: int = 0,
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
        assert R.shape == (self.num_states, self.num_actions, self.num_states)

        uniform = np.ones(self.num_actions) / self.num_actions
        T_arr     = T(uniform)       if callable(T)       else T
        B_arr     = B(uniform)       if callable(B)       else B
        th0_arr   = theta_0(uniform) if callable(theta_0) else theta_0

        assert T_arr.shape   == (self.num_states, self.num_actions, self.num_states), \
            f"T shape {T_arr.shape} != ({self.num_states},{self.num_actions},{self.num_states})"
        assert B_arr.shape   == (self.num_states, self.num_obs), \
            f"B shape {B_arr.shape} != ({self.num_states},{self.num_obs})"
        assert th0_arr.shape == (self.num_states,), \
            f"theta_0 shape {th0_arr.shape} != ({self.num_states},)"
        assert np.allclose(T_arr.sum(axis=2), 1),  "T rows must sum to 1"
        assert np.allclose(B_arr.sum(axis=1), 1),  "B rows must sum to 1"
        assert np.isclose(th0_arr.sum(), 1),        "theta_0 must sum to 1"

        return (T, B, theta_0, R)

    def mix_params(self, params_list: list, coefficients: np.ndarray):
        """Mix point-valued POMDP hypotheses into a credal mixture.

        Returns list of (params, prior_weight) tuples. Flattens nested mixtures.
        """
        components = []
        for p, c in zip(params_list, coefficients):
            if isinstance(p, list):          # already a mixture — flatten
                components.extend((cp, w * float(c)) for cp, w in p)
            else:
                components.append((p, float(c)))
        total = sum(w for _, w in components)
        return [(cp, w / total) for cp, w in components]

    def event_index(self, outcome: Outcome, action: int) -> int:
        return outcome.observation

    # ── Belief state ──────────────────────────────────────────────────────────

    def initial_state(self):
        return None

    def is_initial(self, state) -> bool:
        return state is None

    def _initial_belief(self, component_params, policy=None) -> np.ndarray:
        """Initial belief for one component, resolving θ₀ against policy."""
        _T, _B, theta_0_raw, _R = component_params
        return self._resolve(theta_0_raw, policy).copy()

    # ── Bayesian filter ───────────────────────────────────────────────────────

    def update_state(self, state, outcome: Outcome, action: int,
                     policy: np.ndarray, params=None):
        """Standard POMDP Bayesian filter applied per component.

        State: list[(belief_vec, log_marginal)], one per component.
        Lazily initialises from θ₀ when state is None.
        """
        components = params if isinstance(params, list) else [(params, 1.0)]
        if state is None:
            state = [(self._initial_belief(cp, policy), 0.0)
                     for cp, _w in components]

        obs_idx = outcome.observation
        new_state = []
        for (belief, log_m), (component_params, _w) in zip(state, components):
            T_raw, B_raw, _theta_0, _R = component_params
            T_arr = self._resolve(T_raw, policy)
            B_arr = self._resolve(B_raw, policy)

            belief_pred = belief @ T_arr[:, action, :]   # shape (|S|,)
            belief_post = B_arr[:, obs_idx] * belief_pred
            total = belief_post.sum()

            if total <= 0:
                new_state.append((belief.copy(), log_m + np.log(1e-300)))
            else:
                new_state.append((belief_post / total, log_m + np.log(total)))

        return new_state

    def _posterior_weights(self, state, params) -> np.ndarray:
        """Posterior credal weights: weights[k] ∝ prior_weight[k] · P(history|k)."""
        log_marginals  = np.array([lm for _b, lm in state])
        prior_log_w    = np.log([w  for _,  w in params])
        log_w = prior_log_w + log_marginals
        log_w -= log_w.max()
        w = np.exp(log_w)
        return w / w.sum()

    # ── IB likelihood (one-step, used by Infradistribution.update) ───────────

    def compute_likelihood(self, belief_state, outcome: Outcome, params,
                           action: int, policy=None) -> float:
        """P(observation | belief, action) averaged over components by posterior weights."""
        components = params if isinstance(params, list) else [(params, 1.0)]
        beliefs = (
            [(self._initial_belief(cp, policy), 0.0) for cp, _w in components]
            if belief_state is None else belief_state
        )
        weights  = self._posterior_weights(beliefs, components)
        obs_idx  = outcome.observation
        total    = 0.0
        for (belief, _lm), (cp, _w), weight in zip(beliefs, components, weights):
            T_arr = self._resolve(cp[0], policy)
            B_arr = self._resolve(cp[1], policy)
            belief_pred = belief @ T_arr[:, action, :]
            total += weight * float(belief_pred @ B_arr[:, obs_idx])
        return total

    def compute_expected_reward(self, belief_state, reward_function: np.ndarray,
                                params, action: int, policy=None) -> float:
        """E[reward_function[next_obs] | belief, action] — one-step expectation.

        Used by Infradistribution.update for IB normalisation (the gluing
        operator). Does NOT run value iteration. See compute_q_values for
        multi-step planning.
        """
        components = params if isinstance(params, list) else [(params, 1.0)]
        beliefs = (
            [(self._initial_belief(cp, policy), 0.0) for cp, _w in components]
            if belief_state is None else belief_state
        )
        weights = self._posterior_weights(beliefs, components)
        total   = 0.0
        for (belief, _lm), (cp, _w), weight in zip(beliefs, components, weights):
            T_arr = self._resolve(cp[0], policy)
            B_arr = self._resolve(cp[1], policy)
            # E[rf[next_obs]] = b @ T[:,a,:] @ B @ rf
            belief_pred = belief @ T_arr[:, action, :]          # (|S|,)
            total += weight * float(belief_pred @ (B_arr @ reward_function))
        return total

    # ── Planning (multi-step, used by SupraPOMDPAgent._expected_rewards) ─────

    def compute_q_values(self, belief_state, params,
                         policy: np.ndarray | None = None) -> np.ndarray:
        """Multi-step Q-values Q(b, a) for each action under policy π.

        Runs value iteration per component, weighted by posterior credal weights.
        The candidate policy π is threaded into policy-dependent kernels so
        that Newcomb-family predictors see the correct committed policy.

        Returns np.ndarray of shape (num_actions,).
        """
        components = params if isinstance(params, list) else [(params, 1.0)]
        beliefs = (
            [(self._initial_belief(cp, policy), 0.0) for cp, _w in components]
            if belief_state is None else belief_state
        )
        weights     = self._posterior_weights(beliefs, components)
        policy_key  = None if policy is None else policy.tobytes()
        Q_total     = np.zeros(self.num_actions)

        for (belief, _lm), (cp, _w), credal_weight in zip(beliefs, components, weights):
            T_raw = cp[0]; R = cp[3]
            T_arr = self._resolve(T_raw, policy)

            cache_key = (
                None if policy_key is None else (id(cp), policy_key)
            )
            V_s = self._value_iteration(T_arr, R, policy, cache_key=cache_key)

            for a in range(self.num_actions):
                # Q(b,a) = Σ_s b[s] · Σ_{s'} T[s,a,s'] · (R[s,a,s'] + γ·V[s'])
                R_sa = (T_arr[:, a, :] * R[:, a, :]).sum(axis=1)  # (|S|,)
                EV   = T_arr[:, a, :] @ V_s                        # (|S|,)
                Q_total[a] += credal_weight * float(
                    belief @ (R_sa + self.discount * EV)
                )

        return Q_total

    def _value_iteration(self, T: np.ndarray, R: np.ndarray,
                         policy: np.ndarray, cache_key=None) -> np.ndarray:
        """Standard infinite-horizon value iteration under fixed stochastic policy π.

        V[s] = Σ_a π[a] · Σ_{s'} T[s,a,s'] · (R[s,a,s'] + γ·V[s'])

        Returns shape (|S|,). Caches result under cache_key if provided.
        """
        if cache_key is not None and cache_key in self._v_cache:
            return self._v_cache[cache_key]

        if policy is None:
            policy = np.ones(self.num_actions) / self.num_actions

        # Pre-compute E_{s'}[R | s, a] — shape (|S|, |A|), independent of V
        R_pi = (T * R).sum(axis=2)
        V    = np.zeros(self.num_states)

        for _ in range(self.value_iter_max):
            EV_pi = T @ V                          # (|S|, |A|): E[V | s, a]
            Q     = R_pi + self.discount * EV_pi   # (|S|, |A|)
            V_new = Q @ policy                     # (|S|,)
            if np.max(np.abs(V_new - V)) < self.value_iter_tol:
                V = V_new
                break
            V = V_new

        if cache_key is not None:
            self._v_cache[cache_key] = V
        return V
