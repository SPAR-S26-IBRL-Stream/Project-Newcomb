import numpy as np
import pytest
from ibrl.agents.supra_pomdp_agent import StatefulPolicy
from ibrl.utils.belief_discretization import BeliefIndexer, simplex_grid, corner_beliefs


class TestStatefulPolicyConstruction:
    def test_initialization(self):
        num_beliefs, num_actions = 3, 2
        policy_table = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        
        def dummy_indexer(belief):
            return 0
        
        policy = StatefulPolicy(policy_table, dummy_indexer)
        assert policy.num_beliefs == 3
        assert policy.num_actions == 2
    
    def test_uniform_policy(self):
        num_beliefs, num_actions = 4, 3
        indexer = BeliefIndexer(corner_beliefs(3))
        policy = StatefulPolicy.uniform(num_beliefs, num_actions, indexer)
        
        # Each entry should be 1/num_actions
        expected_val = 1.0 / num_actions
        assert np.allclose(policy.policy_table, expected_val)
    
    def test_call_interface(self):
        """Policy should be callable."""
        policy_table = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        
        indexer = BeliefIndexer(np.array([[1, 0], [0, 1]]))
        policy = StatefulPolicy(policy_table, indexer)
        
        belief_0 = np.array([1.0, 0.0])
        result = policy(belief_0)
        np.testing.assert_array_almost_equal(result, [0.8, 0.2])
    
    def test_to_flat_policy(self):
        """Averaging over beliefs gives flat policy."""
        policy_table = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        indexer = BeliefIndexer(np.array([[1, 0], [0, 1]]))
        policy = StatefulPolicy(policy_table, indexer)
        
        flat = policy.to_flat_policy()
        expected = np.array([0.55, 0.45])  # mean of each column
        np.testing.assert_array_almost_equal(flat, expected)


class TestBeliefDiscretization:
    def test_corner_beliefs(self):
        beliefs = corner_beliefs(3)
        assert beliefs.shape == (3, 3)
        np.testing.assert_array_almost_equal(beliefs, np.eye(3))
    
    def test_simplex_grid_2d(self):
        beliefs = simplex_grid(2, 3)
        # For 2D: [1,0], [0.5, 0.5], [0, 1]
        assert all(np.isclose(b.sum(), 1.0) for b in beliefs)
        assert all(np.all(b >= 0) for b in beliefs)
    
    def test_belief_indexer(self):
        centers = np.array([[1, 0], [0, 1], [0.5, 0.5]])
        indexer = BeliefIndexer(centers)
        
        # Belief close to first center
        idx = indexer(np.array([0.99, 0.01]))
        assert idx == 0
        
        # Belief close to third center
        idx = indexer(np.array([0.5, 0.5]))
        assert idx == 2


class TestPolicyOptimization:
    def test_greedy_policy_extraction(self):
        """Greedy policy selects best action per belief."""
        Q_table = np.array([
            [1.0, 2.0, 0.5],  # Belief 0: prefer action 1
            [3.0, 1.0, 0.5],  # Belief 1: prefer action 0
            [0.5, 0.5, 2.0]   # Belief 2: prefer action 2
        ])
        
        indexer = BeliefIndexer(corner_beliefs(3))
        from ibrl.agents.policy_optimizer import PolicyOptimizer
        
        optimizer = PolicyOptimizer(Q_table, indexer, num_actions=3)
        policy = optimizer.greedy_policy()
        
        # Check greediness
        assert policy[0, 1] == 1.0  # Action 1 has Q=2.0
        assert policy[1, 0] == 1.0  # Action 0 has Q=3.0
        assert policy[2, 2] == 1.0  # Action 2 has Q=2.0
    
    def test_softmax_policy_valid_distribution(self):
        """Softmax policy returns valid probability distributions."""
        Q_table = np.array([
            [1.0, 2.0],
            [0.5, 0.5]
        ])
        
        indexer = BeliefIndexer(corner_beliefs(2))
        from ibrl.agents.policy_optimizer import PolicyOptimizer
        
        optimizer = PolicyOptimizer(Q_table, indexer, num_actions=2)
        policy = optimizer.softmax_policy(beta=1.0)
        
        # Check normalization
        for b in range(policy.num_beliefs):
            assert np.isclose(policy[b, :].sum(), 1.0)
            assert np.all(policy[b, :] >= 0)
