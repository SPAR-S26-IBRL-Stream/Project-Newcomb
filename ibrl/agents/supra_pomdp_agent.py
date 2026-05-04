from dataclasses import dataclass
import numpy as np
from typing import Callable, Tuple


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
        


## Code rationale :

# captured belief-indexed action distributions
# Provides with both direct indexing and callable interface
