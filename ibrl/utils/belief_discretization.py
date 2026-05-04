"""Utilities for discretizing belief spaces."""
import numpy as np
from typing import Callable, Tuple
from scipy.spatial.distance import euclidean


class BeliefIndexer:
    """Maps continuous beliefs to discrete indices."""
    
    def __init__(self, belief_centers: np.ndarray, belief_tol: float = 1e-6):
        """
        Args:
            belief_centers: Array of shape (num_beliefs, num_states) 
                           representing discretized belief points
            belief_tol: Tolerance for nearest-neighbor matching
        """
        self.belief_centers = belief_centers
        self.belief_tol = belief_tol
        self.num_beliefs = belief_centers.shape[0]
        self.num_states = belief_centers.shape[1]
    
    def __call__(self, belief: np.ndarray) -> int:
        """Find nearest belief center index."""
        if belief.shape != (self.num_states,):
            raise ValueError(f"Belief shape {belief.shape} != ({self.num_states},)")
        
        distances = np.array([euclidean(belief, center) 
                             for center in self.belief_centers])
        best_idx = np.argmin(distances)
        
        # Warn if belief is far from any center (poor discretization)
        if distances[best_idx] > self.belief_tol:
            import warnings
            warnings.warn(f"Belief {belief} is far from nearest center "
                         f"(distance={distances[best_idx]:.4f})")
        
        return int(best_idx)


def simplex_grid(num_states: int, num_points_per_dim: int) -> np.ndarray:
    """
    Generate grid of beliefs uniformly distributed on probability simplex.
    
    For num_states=2, num_points=3 → beliefs [1,0], [0.5,0.5], [0,1]
    For num_states=3, num_points=3 → 6 points on 2-simplex
    
    Returns:
        Array of shape (num_beliefs, num_states) with each row a valid belief (sums to 1)
    """
    if num_states == 1:
        return np.array([[1.0]])
    
    # Use itertools to enumerate all ways to distribute num_points_per_dim
    # across num_states dimensions
    import itertools
    
    beliefs = []
    for combo in itertools.product(range(num_points_per_dim), repeat=num_states-1):
        # combo represents first num_states-1 components
        first_parts = np.array(combo, dtype=float) / (num_points_per_dim - 1)
        last_part = 1.0 - first_parts.sum()
        
        if 0 <= last_part <= 1:  # Valid belief
            belief = np.concatenate([first_parts, [last_part]])
            beliefs.append(belief)
    
    # Remove duplicates
    beliefs = np.array(beliefs)
    beliefs = np.unique(np.round(beliefs, decimals=10), axis=0)
    return beliefs


def corner_beliefs(num_states: int) -> np.ndarray:
    """
    Generate corner beliefs (maximum certainty states).
    For num_states=3: [[1,0,0], [0,1,0], [0,0,1]]
    """
    return np.eye(num_states)
    
    
    
    
    
    
    
    
    
    
    
## Rationale:

# Helps to facilitate discretization of continuous belief space
# Maintains a Simplex grid: systematic coverage of probability space
# Corner beliefs: useful baseline for some domains
# Nearest-neighbor matching: practical for continuous beliefs
