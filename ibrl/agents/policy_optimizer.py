"""Policy optimization for belief-indexed POMDPs."""
import numpy as np
from typing import Tuple, Callable
import scipy.optimize as opt


class PolicyOptimizer:
    """Optimizes belief-dependent policies given Q-values."""
    
    def __init__(self, Q_table: np.ndarray, belief_indexer, num_actions: int):
        """
        Args:
            Q_table: Shape (num_beliefs, num_actions), Q_table[b,a] = expected value
            belief_indexer: Maps beliefs to table indices
            num_actions: Number of actions
        """
        self.Q_table = Q_table
        self.belief_indexer = belief_indexer
        self.num_actions = num_actions
        self.num_beliefs = Q_table.shape[0]
    
    def greedy_policy(self) -> "StatefulPolicy":
        """
        Greedy policy: π(b) = e_{argmax_a Q(b,a)}
        Returns deterministic policy (one-hot for each belief).
        """
        from ..agents.supra_pomdp_agent import StatefulPolicy
        
        policy_table = np.zeros((self.num_beliefs, self.num_actions))
        for b_idx in range(self.num_beliefs):
            best_action = np.argmax(self.Q_table[b_idx, :])
            policy_table[b_idx, best_action] = 1.0
        
        return StatefulPolicy(policy_table, self.belief_indexer)
    
    def softmax_policy(self, beta: float = 1.0) -> "StatefulPolicy":
        """
        Softmax policy: π(b, a) = exp(β·Q(b,a)) / Z(b)
        
        Args:
            beta: Inverse temperature. High β → more deterministic.
        """
        from ..agents.supra_pomdp_agent import StatefulPolicy
        
        policy_table = np.zeros((self.num_beliefs, self.num_actions))
        
        for b_idx in range(self.num_beliefs):
            q_vals = self.Q_table[b_idx, :]
            # Numerical stability: subtract max
            q_shifted = q_vals - q_vals.max()
            policy_table[b_idx, :] = np.exp(beta * q_shifted)
            policy_table[b_idx, :] /= policy_table[b_idx, :].sum()
        
        return StatefulPolicy(policy_table, self.belief_indexer)
    
    def linear_program_policy(self) -> "StatefulPolicy":
        """
        Optimal policy via LP (for acyclic POMDP value functions).
        
        Solves: max_π [ Σ_b ρ(b) Σ_a π(b,a) Q(b,a) ]
        s.t. Σ_a π(b,a) = 1 for all b;
             π(b,a) ≥ 0 (usual conditions given actions and beliefs)
        
        Where ρ(b) is the stationary belief distribution (approximated uniformly).
        
        Returns:
            Optimal belief-dependent policy
        """
        from ..agents.supra_pomdp_agent import StatefulPolicy
        
        # Flatten policy: [π(b=0,a=0), π(b=0,a=1), ..., π(b=B,a=A)]
        num_vars = self.num_beliefs * self.num_actions
        
        # Objective: maximize Σ Q(b,a) * π(b,a)
        # Linear program maximizes c^T x
        c = -self.Q_table.flatten()  # Negative for maximization in scipy.optimize
        
        # Constraints: Σ_a π(b,a) = 1 for each b
        A_eq = []
        b_eq = []
        for b_idx in range(self.num_beliefs):
            constraint = np.zeros(num_vars)
            constraint[b_idx * self.num_actions : (b_idx + 1) * self.num_actions] = 1.0
            A_eq.append(constraint)
            b_eq.append(1.0)
        
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        
        # Bounds: 0 ≤ π(b,a) ≤ 1
        bounds = [(0, 1) for _ in range(num_vars)]
        
        # Solve
        result = opt.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not result.success:
            raise RuntimeError(f"LP failed: {result.message}")
        
        policy_table = result.x.reshape((self.num_beliefs, self.num_actions))
        return StatefulPolicy(policy_table, self.belief_indexer)
        
      
      
## Code Rationale:

# Three policy extraction methods with increasing optimality( as follows)
# Greedy: simplest, may be suboptimal for stochastic POMDPs
# Softmax: soft approximation, useful for exploration
# LP: theoretically optimal for linear value functions
# All return standardized StatefulPolicy objects
