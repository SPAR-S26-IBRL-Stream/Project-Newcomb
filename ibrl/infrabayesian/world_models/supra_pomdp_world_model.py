"""SupraPOMDPWorldModel — world model for finite-state POMDPs with optional policy-dependence."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .base import WorldModel
from ...outcome import Outcome

def to_str(self, params) -> str:
    return "[SupraPOMDP]"


@dataclass
class SupraPOMDPWorldModelBeliefState:
    """Belief state: list of (belief_vec, log_marginal) per component, or None before first update."""
    components: list[tuple[np.ndarray, float]] | None


class SupraPOMDPWorldModel(WorldModel):
    """
    World model for stateful POMDPs with optional policy-dependent kernels.
    
    Supports finite credal mixtures of point-valued POMDPs. Each hypothesis is
    (T, B, theta_0, R) where T, B, theta_0 may optionally be callables of policy.
    Belief state is per-component (belief_vec, log_marginal) tracking.
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
        self._v_cache: dict = {}

    @staticmethod
    def _resolve(kernel_or_fn, policy):
        """Resolve kernel: if callable, call with policy; else return as-is."""
        return kernel_or_fn(policy) if callable(kernel_or_fn) else kernel_or_fn

    def make_params(self, T, B, theta_0, R: np.ndarray):
        """Validate and bundle POMDP parameters. T, B, theta_0 may be static arrays or callables."""
        assert R.shape == (self.num_states, self.num_actions, self.num_states)
        
        # Validate by resolving at uniform policy
        uniform = np.ones(self.num_actions) / self.num_actions
        T_arr = self._resolve(T, uniform)
        B_arr = self._resolve(B, uniform)
        theta_0_arr = self._resolve(theta_0, uniform)
        
        assert T_arr.shape == (self.num_states, self.num_actions, self.num_states)
        assert B_arr.shape == (self.num_states, self.num_obs)
        assert theta_0_arr.shape == (self.num_states,)
        assert np.allclose(T_arr.sum(axis=2), 1), "T rows must sum to 1"
        assert np.allclose(B_arr.sum(axis=1), 1), "B rows must sum to 1"
        assert np.isclose(theta_0_arr.sum(), 1), "theta_0 must sum to 1"
        
        return (T, B, theta_0, R)

    def mix_params(self, params_list: list, coefficients: np.ndarray):
        """Mix POMDP hypotheses into credal mixture. Returns list of (params, weight) tuples."""
        components = []
        for p, c in zip(params_list, coefficients):
            if isinstance(p, list):  # Flatten nested mixtures
                components.extend((cp, w * float(c)) for cp, w in p)
            else:
                components.append((p, float(c)))
        total = sum(w for _, w in components)
        return [(cp, w / total) for cp, w in components]

    def event_index(self, outcome: Outcome, action: int) -> int:
        """Extract observation index from outcome."""
        return outcome.observation

    def initial_state(self) -> SupraPOMDPWorldModelBeliefState:
        """Initial belief state is None; lazily initialized on first update."""
        return SupraPOMDPWorldModelBeliefState(None)

    def is_initial(self, state: SupraPOMDPWorldModelBeliefState) -> bool:
        """Check if state is initial (uninitialized)."""
        return state.components is None

    def _initial_belief(self, component_params, policy=None):
        """Construct initial belief for one component, resolving theta_0 against policy."""
        _T, _B, theta_0_raw, _R = component_params
        return self._resolve(theta_0_raw, policy).copy()

    def _posterior_weights(self, state: SupraPOMDPWorldModelBeliefState,
                          params: list) -> np.ndarray:
        """Compute posterior credal weights: prior_weight[k] * P(history | k)."""
        log_marginals = np.array([lm for (_b, lm) in state.components])
        prior_log_w = np.log([w for _, w in params])
        log_w = prior_log_w + log_marginals
        log_w -= log_w.max()
        w = np.exp(log_w)
        return w / w.sum()

    def update_state(self, state: SupraPOMDPWorldModelBeliefState, outcome: Outcome,
                     action: int, policy: np.ndarray, params=None) -> SupraPOMDPWorldModelBeliefState:
        """Standard POMDP Bayesian filter applied per-component."""
        components = params if isinstance(params, list) else [(params, 1.0)]
        
        if state.components is None:
            state = SupraPOMDPWorldModelBeliefState(
                [(self._initial_belief(c[0], policy), 0.0) for c in components]
            )
        
        new_state = []
        obs = outcome.observation
        for (belief, log_m), (component_params, _w) in zip(state.components, components):
            T_raw, B_raw, _theta_0, _R = component_params
            T_arr = self._resolve(T_raw, policy)
            B_arr = self._resolve(B_raw, policy)
            
            belief_pred = belief @ T_arr[:, action, :]
            belief_post = B_arr[:, obs] * belief_pred
            total = belief_post.sum()
            
            if total <= 0:
                new_state.append((belief.copy(), log_m + np.log(1e-300)))
            else:
                new_state.append((belief_post / total, log_m + np.log(total)))
        
        return SupraPOMDPWorldModelBeliefState(new_state)

    def compute_likelihood(self, belief_state: SupraPOMDPWorldModelBeliefState,
                          outcome: Outcome, params, action: int, policy) -> float:
        """P(observation | belief, action) averaged over components by posterior weights."""
        components = params if isinstance(params, list) else [(params, 1.0)]
        
        if belief_state.components is None:
            beliefs = SupraPOMDPWorldModelBeliefState(
                [(self._initial_belief(c[0], policy), 0.0) for c in components]
            )
        else:
            beliefs = belief_state
        
        weights = self._posterior_weights(beliefs, components)
        obs = outcome.observation
        total = 0.0
        
        for (belief, _lm), (component_params, _w), weight in zip(
            beliefs.components, components, weights
        ):
            T_raw, B_raw, _, _ = component_params
            T_arr = self._resolve(T_raw, policy)
            B_arr = self._resolve(B_raw, policy)
            belief_pred = belief @ T_arr[:, action, :]
            total += weight * float((belief_pred @ B_arr[:, obs]))
        
        return total

    def compute_expected_reward(self, belief_state: SupraPOMDPWorldModelBeliefState,
                               reward_function: np.ndarray, params, action: int,
                               policy) -> float:
        """Multi-step expected reward via value iteration, averaged over components."""
        components = params if isinstance(params, list) else [(params, 1.0)]
        
        if belief_state.components is None:
            beliefs = SupraPOMDPWorldModelBeliefState(
                [(self._initial_belief(c[0], policy), 0.0) for c in components]
            )
        else:
            beliefs = belief_state
        
        weights = self._posterior_weights(beliefs, components)
        total = 0.0
        policy_key = None if policy is None else policy.tobytes()
        
        for (belief, _lm), (component_params, _w), credal_weight in zip(
            beliefs.components, components, weights
        ):
            T_raw, _B, _theta_0, R = component_params
            T_arr = self._resolve(T_raw, policy)
            cache_key = None if policy_key is None else (id(component_params), policy_key)
            V_s = self._value_iteration(T_arr, R, policy, cache_key=cache_key)
            
            R_sa = (T_arr[:, action, :] * R[:, action, :]).sum(axis=1)
            EV = T_arr[:, action, :] @ V_s
            Q_sa = R_sa + self.discount * EV
            total += credal_weight * float(belief @ Q_sa)
        
        return total

    def _value_iteration(self, T: np.ndarray, R: np.ndarray,
                        policy: np.ndarray, cache_key=None) -> np.ndarray:
        """Infinite-horizon value iteration under fixed stochastic policy."""
        if cache_key is not None and cache_key in self._v_cache:
            return self._v_cache[cache_key]
        
        if policy is None:
            policy = np.ones(self.num_actions) / self.num_actions
        
        R_pi = (T * R).sum(axis=2)
        V = np.zeros(self.num_states)
        
        for _ in range(self.value_iter_max):
            EV_pi = T @ V
            Q = R_pi + self.discount * EV_pi
            V_new = Q @ policy
            if np.max(np.abs(V_new - V)) < self.value_iter_tol:
                V = V_new
                break
            V = V_new
        
        if cache_key is not None:
            self._v_cache[cache_key] = V
        
        return V
    
    def compute_q_values_belief_indexed(
        self, 
        belief_state: SupraPOMDPWorldModelBeliefState,
        params,
        policy_discretisation: int = 0,
        belief_space: np.ndarray = None
    ) -> Tuple[np.ndarray, BeliefIndexer]:
        """
        Compute Q(b, a) for each belief b in discretized space and each action a.
        
        This enables the agent to optimize over belief-dependent policies.
        
        Args:
            belief_state: Current belief (unused for Q-table computation)
            params: World model parameters
            policy_discretisation: Number of belief points per dimension
                                  0 = compute only for corner beliefs
            belief_space: Explicit array of beliefs (shape (num_beliefs, num_states))
                         If None, generated via simplex_grid()
        
        Returns:
            Q_table: Array of shape (num_beliefs, num_actions)
                    Q_table[b, a] = expected value of taking action a in belief b
            belief_indexer: Function mapping beliefs to indices
        
        Key insight:
            Once we have Q(b, a), we can optimize via:
            π*(b) = argmax_a Q(b, a)  (greedy)
            or π*(b) = softmax(β · Q(b, :))  (soft)
        """
        if policy_discretisation == 0:
            # Use only corner beliefs (fully observable states)
            belief_space = np.eye(self.num_states)
        elif belief_space is None:
            # Generate simplex grid
            from ...utils.belief_discretization import simplex_grid
            belief_space = simplex_grid(self.num_states, policy_discretisation)
        
        num_beliefs = belief_space.shape[0]
        Q_table = np.zeros((num_beliefs, self.num_actions))
        
        # For each discretized belief, compute Q values
        for b_idx, belief_point in enumerate(belief_space):
            # Create a synthetic belief state at this point
            components = params if isinstance(params, list) else [(params, 1.0)]
            synthetic_state = SupraPOMDPWorldModelBeliefState(
                [(belief_point.copy(), 0.0) for _c in components]
            )
            
            # Compute Q values for each action at this belief
            for a in range(self.num_actions):
                # Use existing value iteration machinery with flat policy
                # (we'll optimize the policy separately)
                uniform_policy = np.ones(self.num_actions) / self.num_actions
                
                weights = self._posterior_weights(synthetic_state, components)
                Q_a = 0.0
                
                for (belief, _lm), (component_params, _w), credal_weight in zip(
                    synthetic_state.components, components, weights
                ):
                    T_raw, _B, _theta_0, R = component_params
                    T_arr = self._resolve(T_raw, uniform_policy)
                    
                    # One-step reward
                    R_sa = (T_arr[:, a, :] * R[:, a, :]).sum(axis=1)
                    one_step = belief @ R_sa
                    
                    # Future value: V(b') after taking action a
                    belief_next = belief @ T_arr[:, a, :]
                    belief_next = belief_next / (belief_next.sum() + 1e-300)
                    
                    cache_key = None
                    V_s = self._value_iteration(T_arr, R, uniform_policy, cache_key=cache_key)
                    future = belief_next @ (self.discount * V_s)
                    
                    Q_a += credal_weight * (one_step + future)
                
                Q_table[b_idx, a] = Q_a
        
        # Create belief indexer
        from ...utils.belief_discretization import BeliefIndexer
        belief_indexer = BeliefIndexer(belief_space, belief_tol=1e-6)
        
        return Q_table, belief_indexer
     
     
     
    def compute_q_values(self, belief_state, params, policy):
        """Compute Q-values for all actions at current belief."""
        Q = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            Q[a] = self.compute_expected_reward(belief_state, np.ones(self.num_obs), 
                                               params, action=a, policy=policy)
        return Q
            
        
        
        
        
        
        
        
        
        
        
## Code Rationale:

# Computes action values for each discretized belief
# Foundation for belief-dependent policy optimization

