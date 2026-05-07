"""SupraPOMDPAgent — InfraBayesianAgent specialized for SupraPOMDP hypotheses."""
from dataclasses import dataclass
import numpy as np
from typing import Callable

from .infrabayesian import InfraBayesianAgent


@dataclass
class StatefulPolicy:
    """
    Belief-indexed policy π: B → Δ(A).
    
    For finite obs spaces, discretizes observation history into belief indices.
    For manageable cases, stores action distributions per belief.
    
    Storage: policy_table[belief_idx] = action_distribution (shape |A|)
    """
    policy_table: np.ndarray  # shape (num_beliefs, num_actions)
    belief_indexer: Callable[[np.ndarray], int]  # Maps belief → index in policy_table
    
    def __call__(self, belief: np.ndarray) -> np.ndarray:
        """Get action distribution for given belief."""
        idx = self.belief_indexer(belief)
        return self.policy_table[idx].copy()
    
    def __getitem__(self, belief_idx: int) -> np.ndarray:
        """Direct access by belief index."""
        return self.policy_table[belief_idx]
    
    def set_action_dist(self, belief_idx: int, action_dist: np.ndarray):
        """Set action distribution for a belief."""
        assert np.isclose(action_dist.sum(), 1.0)
        self.policy_table[belief_idx] = action_dist

    @property
    def num_actions(self) -> int:
        return self.policy_table.shape[1]
    
    @property
    def num_beliefs(self) -> int:
        return self.policy_table.shape[0]

    def to_flat_policy(self) -> np.ndarray:
        """Convert to old flat policy (average over beliefs, for backward compat)."""
        return self.policy_table.mean(axis=0)

    @classmethod
    def uniform(cls, num_beliefs: int, num_actions: int, 
                belief_indexer: Callable) -> "StatefulPolicy":
        """Create uniform policy over all beliefs and actions."""
        policy_table = np.ones((num_beliefs, num_actions)) / num_actions
        return cls(policy_table, belief_indexer)


class SupraPOMDPAgent(InfraBayesianAgent):
    """
    InfraBayesianAgent specialized for SupraPOMDP hypotheses.
    
    Computes belief-dependent policies by:
    1. Discretizing the belief space
    2. Computing Q-values for each (belief, action) pair
    3. Optimizing a policy that conditions on belief
    
    The policy is recomputed after each update to reflect learned beliefs.
    """
    
    def __init__(self, 
                 num_actions: int,
                 hypotheses: list,
                 prior: np.ndarray,
                 reward_function: np.ndarray,
                 policy_discretisation: int = 10,
                 policy_optimization: str = "greedy",
                 softmax_beta: float = 1.0,
                 exploration_prefix: int = 0,
                 seed: int = 42,
                 **kwargs):
        
        super().__init__(
            num_actions=num_actions,
            hypotheses=hypotheses,
            prior=prior,
            reward_function=reward_function,
            policy_discretisation=policy_discretisation,
            policy_optimization=policy_optimization,
            softmax_beta=softmax_beta,
            exploration_prefix=exploration_prefix,
            seed=seed,
            **kwargs
        )
    
    def get_probabilities(self) -> np.ndarray:
        """
        For SupraPOMDP with policy-dependent kernels, optimize over pure policies.
        
        For Transparent Newcomb: θ₀(π) depends on π, so we must evaluate
        each candidate policy against the actual predictor response to that policy.
        """
        if self.current_stateful_policy is not None:
            return self.current_stateful_policy.to_flat_policy()
        
        # For policy-dependent models, evaluate pure policies
        # (one-hot distributions over actions)
        pure_policies = np.eye(self.num_actions)
        rewards = []
        
        for pure_policy in pure_policies:
            # Evaluate this pure policy against all actions
            action_values = np.array([
                self.dist.evaluate_action(self.reward_function[a], a, pure_policy)
                for a in range(self.num_actions)
            ])
            # Expected value of this policy: Σ_a π(a) * E[a]
            policy_value = pure_policy @ action_values
            rewards.append(policy_value)
        
        rewards = np.array(rewards)
        return self.build_greedy_policy(rewards)



