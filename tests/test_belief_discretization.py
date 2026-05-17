"""Tests for belief discretization and stateful policy utilities."""
import numpy as np
import pytest
from ibrl.utils.belief_discretization import BeliefIndexer, simplex_grid, corner_beliefs
from ibrl.agents.supra_pomdp_agent import BeliefPolicy


class TestBeliefDiscretization:
    def test_corner_beliefs(self):
        beliefs = corner_beliefs(3)
        assert beliefs.shape == (3, 3)
        np.testing.assert_array_almost_equal(beliefs, np.eye(3))

    def test_simplex_grid_2d(self):
        beliefs = simplex_grid(2, 3)
        assert all(np.isclose(b.sum(), 1.0) for b in beliefs)
        assert all(np.all(b >= 0) for b in beliefs)

    def test_belief_indexer(self):
        centers = np.array([[1, 0], [0, 1], [0.5, 0.5]])
        indexer = BeliefIndexer(centers)
        
        idx = indexer(np.array([0.99, 0.01]))
        assert idx == 0
        
        idx = indexer(np.array([0.5, 0.5]))
        assert idx == 2


class TestBeliefPolicy:
    def test_initialization(self):
        belief_points = np.array([[1, 0], [0, 1]])
        policy_table = np.array([[1.0, 0.0], [0.0, 1.0]])
        policy = BeliefPolicy(belief_points, policy_table)
        assert policy.belief_points.shape == (2, 2)
        assert policy.policy_table.shape == (2, 2)

    def test_action_dist_flat_policy(self):
        """Flat policy returns same distribution for any belief."""
        belief_points = np.array([[1.0]])
        policy_table = np.array([[0.6, 0.4]])
        policy = BeliefPolicy(belief_points, policy_table)
        
        result = policy.action_dist(np.array([0.5, 0.5]))
        np.testing.assert_array_almost_equal(result, [0.6, 0.4])

    def test_action_dist_belief_indexed(self):
        """Belief-indexed policy returns different distributions per belief."""
        belief_points = np.array([[1, 0], [0, 1], [0.5, 0.5]])
        policy_table = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.5, 0.5]
        ])
        indexer = BeliefIndexer(belief_points)
        policy = BeliefPolicy(belief_points, policy_table, indexer)
        
        result_0 = policy.action_dist(np.array([0.99, 0.01]))
        np.testing.assert_array_almost_equal(result_0, [0.8, 0.2])
        
        result_1 = policy.action_dist(np.array([0.01, 0.99]))
        np.testing.assert_array_almost_equal(result_1, [0.3, 0.7])

    def test_cache_key(self):
        """Cache key is hashable and consistent."""
        belief_points = np.array([[1, 0], [0, 1]])
        policy_table = np.array([[0.8, 0.2], [0.3, 0.7]])
        policy = BeliefPolicy(belief_points, policy_table)
        
        key1 = policy.cache_key()
        key2 = policy.cache_key()
        assert key1 == key2
        assert isinstance(key1, bytes)

    def test_flat_constructor(self):
        """Flat policy constructor creates one-row policy."""
        policy = BeliefPolicy.flat(3)
        assert policy.policy_table.shape == (1, 3)
        np.testing.assert_array_almost_equal(policy.policy_table[0], [1/3, 1/3, 1/3])
