"""SupraPOMDPWorldModel — world model for a finite mixture of point-valued POMDPs."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .base import WorldModel
from ...outcome import Outcome


# Representation of hypothesis parameters and belief state of Supra POMDP world model

@dataclass
class SupraPOMDPWorldModelParameters:
    """
    Parameters of Supra POMDP world model: lists of POMDP kernels with prior weights.
        T[k]:       transition for k-th component — shape (|S|,|A|,|S|) or callable policy → that shape
        B[k]:       observation for k-th component — shape (|S|,|O|)      or callable policy → that shape
        theta_0[k]: initial belief for k-th component — shape (|S|,)      or callable policy → that shape
        R[k]:       reward for k-th component — shape (|S|,|A|,|S|), always static
        weights:    prior weights, shape (num_components,)
    """
    T:       list
    B:       list
    theta_0: list
    R:       list
    weights: np.ndarray

@dataclass
class SupraPOMDPWorldModelBeliefState:
    """
    Belief state of Supra POMDP world model: list of (belief_vec, log_marginal) per component,
    or None before the first update (lazy initialization).
    """
    components: list[tuple[np.ndarray, float]] | None


class SupraPOMDPWorldModel(WorldModel):
    """
    World model for a stateful POMDP with optional policy-dependent kernels.

    Belief state: SupraPOMDPWorldModelBeliefState containing list of (belief_vec, log_marginal)
        per hypothesis component, or None before the first update.

    Hypothesis params: SupraPOMDPWorldModelParameters with fields T, B, theta_0, R, weights —
        each a list over components. Single hypothesis via make_params(T, B, theta_0, R).

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
        # Cache: (id(T_k), id(R_k), policy_bytes) → V vector.
        self._v_cache: dict = {}

    @staticmethod
    def _resolve(kernel_or_fn, policy):
        """Evaluate kernel_or_fn at policy if callable, else return as-is."""
        return kernel_or_fn(policy) if callable(kernel_or_fn) else kernel_or_fn

    # ── Param construction ────────────────────────────────────────────────────

    def make_params(self, T, B, theta_0, R: np.ndarray) -> SupraPOMDPWorldModelParameters:
        """Validate shapes and stochasticity, then bundle into a parameter object.

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

        return SupraPOMDPWorldModelParameters([T], [B], [theta_0], [R], np.array([1.0]))

    def mix_params(self, params_list: list[SupraPOMDPWorldModelParameters],
                   coefficients: np.ndarray) -> SupraPOMDPWorldModelParameters:
        """Mix POMDP hypotheses. Flattens nested mixtures."""
        T, B, theta_0, R, weights = [], [], [], [], []
        for p, c in zip(params_list, coefficients):
            T.extend(p.T)
            B.extend(p.B)
            theta_0.extend(p.theta_0)
            R.extend(p.R)
            weights.extend(w * float(c) for w in p.weights)
        weights = np.array(weights)
        weights /= weights.sum()
        return SupraPOMDPWorldModelParameters(T, B, theta_0, R, weights)

    def event_index(self, outcome: Outcome, action: int) -> int:
        return outcome.observation

    # ── Belief state ──────────────────────────────────────────────────────────

    def initial_state(self) -> SupraPOMDPWorldModelBeliefState:
        return SupraPOMDPWorldModelBeliefState(None)

    def is_initial(self, state: SupraPOMDPWorldModelBeliefState) -> bool:
        return state.components is None

    def _initial_belief(self, params: SupraPOMDPWorldModelParameters,
                        policy) -> SupraPOMDPWorldModelBeliefState:
        """Initial belief state for all components, resolving each θ₀ against policy."""
        return SupraPOMDPWorldModelBeliefState([
            (self._resolve(th0_k, policy).copy(), 0.0)
            for th0_k in params.theta_0
        ])

    # ── Bayesian filter ───────────────────────────────────────────────────────

    def update_state(self, state: SupraPOMDPWorldModelBeliefState, outcome: Outcome, action: int,
                     policy: np.ndarray, params: SupraPOMDPWorldModelParameters) -> SupraPOMDPWorldModelBeliefState:
        """Standard POMDP Bayesian filter applied per component.

        State: SupraPOMDPWorldModelBeliefState containing list of (belief_vec, log_marginal) per component.
        Lazily initialises from θ₀ when state.components is None.
        """
        state_components = (self._initial_belief(params, policy).components
                            if state.components is None else state.components)

        obs_idx = outcome.observation
        new_state = []
        for (belief, log_m), T_k, B_k in zip(state_components, params.T, params.B):
            T_arr = self._resolve(T_k, policy)
            B_arr = self._resolve(B_k, policy)

            belief_pred = belief @ T_arr[:, action, :]   # shape (|S|,)
            belief_post = B_arr[:, obs_idx] * belief_pred
            total = belief_post.sum()

            if total <= 0:
                new_state.append((belief.copy(), log_m + np.log(1e-300)))
            else:
                new_state.append((belief_post / total, log_m + np.log(total)))

        return SupraPOMDPWorldModelBeliefState(new_state)

    def _posterior_weights(self, state: SupraPOMDPWorldModelBeliefState,
                           params: SupraPOMDPWorldModelParameters) -> np.ndarray:
        """Posterior weights: weights[k] ∝ prior_weight[k] · P(history|k)."""
        log_marginals = np.array([lm for _b, lm in state.components])
        log_w = np.log(params.weights) + log_marginals
        log_w -= log_w.max()
        w = np.exp(log_w)
        return w / w.sum()

    # ── IB likelihood (one-step, used by Infradistribution.update) ───────────

    def compute_likelihood(self, belief_state: SupraPOMDPWorldModelBeliefState, outcome: Outcome,
                           params: SupraPOMDPWorldModelParameters,
                           action: int, policy) -> float:
        """P(observation | belief, action) averaged over components by posterior weights."""
        rf = np.zeros(self.num_obs)
        rf[outcome.observation] = 1.0
        return self.compute_expected_reward(belief_state, rf, params, action, policy)

    def compute_expected_reward(self, belief_state: SupraPOMDPWorldModelBeliefState,
                                reward_function: np.ndarray,
                                params: SupraPOMDPWorldModelParameters,
                                action: int, policy) -> float:
        """E[reward_function[next_obs] | belief, action] — one-step expectation.

        Used by Infradistribution.update for IB normalisation (the gluing
        operator). Does NOT run value iteration. See compute_q_values for
        multi-step planning.
        """
        beliefs = self._initial_belief(params, policy) if belief_state.components is None else belief_state
        weights = self._posterior_weights(beliefs, params)
        total   = 0.0
        for (belief, _lm), T_k, B_k, weight in zip(beliefs.components, params.T, params.B, weights):
            T_arr = self._resolve(T_k, policy)
            B_arr = self._resolve(B_k, policy)
            # E[rf[next_obs]] = b @ T[:,a,:] @ B @ rf
            belief_pred = belief @ T_arr[:, action, :]          # (|S|,)
            total += weight * float(belief_pred @ (B_arr @ reward_function))
        return total

    # ── Planning (multi-step, used by SupraPOMDPAgent._expected_rewards) ─────

    def compute_q_values(self, belief_state: SupraPOMDPWorldModelBeliefState,
                         params: SupraPOMDPWorldModelParameters,
                         policy: np.ndarray) -> np.ndarray:
        """Multi-step Q-values Q(b, a) for each action under policy π.

        Runs value iteration per component, weighted by posterior mixture weights.
        The candidate policy π is threaded into policy-dependent kernels so
        that Newcomb-family predictors see the correct committed policy.

        Returns np.ndarray of shape (num_actions,).
        """
        beliefs    = self._initial_belief(params, policy) if belief_state.components is None else belief_state
        weights    = self._posterior_weights(beliefs, params)
        policy_key = None if policy is None else policy.tobytes()
        Q_total    = np.zeros(self.num_actions)

        for (belief, _lm), T_k, R_k, weight in zip(beliefs.components, params.T, params.R, weights):
            T_arr = self._resolve(T_k, policy)

            cache_key = None if policy_key is None else (id(T_k), id(R_k), policy_key)
            V_s = self._value_iteration(T_arr, R_k, policy, cache_key=cache_key)

            for a in range(self.num_actions):
                # Q(b,a) = Σ_s b[s] · Σ_{s'} T[s,a,s'] · (R[s,a,s'] + γ·V[s'])
                R_sa = (T_arr[:, a, :] * R_k[:, a, :]).sum(axis=1)  # (|S|,)
                EV   = T_arr[:, a, :] @ V_s                          # (|S|,)
                Q_total[a] += weight * float(belief @ (R_sa + self.discount * EV))

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
