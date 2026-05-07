import numpy as np

from ibrl.agents.policy_optimizer import PolicyOptimizer
from ibrl.agents.supra_pomdp_agent import BeliefPolicy
from ibrl.utils.belief_discretization import BeliefIndexer, corner_beliefs, simplex_grid


def test_belief_policy_queries_nearest_belief_row():
    policy = BeliefPolicy(
        belief_points=np.array([[1.0, 0.0], [0.0, 1.0]]),
        policy_table=np.array([[0.9, 0.1], [0.2, 0.8]]),
    )

    np.testing.assert_allclose(policy.action_dist(np.array([0.95, 0.05])), [0.9, 0.1])
    np.testing.assert_allclose(policy.action_dist(np.array([0.1, 0.9])), [0.2, 0.8])


def test_belief_policy_one_row_represents_flat_policy():
    policy = BeliefPolicy(
        belief_points=np.array([[1.0]]),
        policy_table=np.array([[0.25, 0.75]]),
    )

    np.testing.assert_allclose(policy.action_dist(np.array([1.0])), [0.25, 0.75])
    np.testing.assert_allclose(policy.action_dist(np.array([0.0])), [0.25, 0.75])


def test_belief_policy_rejects_implicit_flattening():
    policy = BeliefPolicy(
        belief_points=np.array([[1.0, 0.0], [0.0, 1.0]]),
        policy_table=np.array([[0.9, 0.1], [0.1, 0.9]]),
    )

    assert not hasattr(policy, "to_flat_policy")


def test_belief_policy_cache_key_depends_on_table_and_belief_points():
    p1 = BeliefPolicy(
        belief_points=np.array([[1.0, 0.0], [0.0, 1.0]]),
        policy_table=np.array([[0.9, 0.1], [0.1, 0.9]]),
    )
    p2 = BeliefPolicy(
        belief_points=np.array([[1.0, 0.0], [0.0, 1.0]]),
        policy_table=np.array([[0.5, 0.5], [0.1, 0.9]]),
    )
    p3 = BeliefPolicy(
        belief_points=np.array([[0.0, 1.0], [1.0, 0.0]]),
        policy_table=np.array([[0.9, 0.1], [0.1, 0.9]]),
    )

    assert p1.cache_key() == p1.cache_key()
    assert p1.cache_key() != p2.cache_key()
    assert p1.cache_key() != p3.cache_key()


def test_belief_discretization_builds_valid_simplex_points():
    beliefs = simplex_grid(num_states=3, num_points_per_dim=3)

    assert beliefs.shape[1] == 3
    np.testing.assert_allclose(beliefs.sum(axis=1), np.ones(beliefs.shape[0]))
    assert np.all(beliefs >= 0)
    assert any(np.allclose(b, [0.5, 0.5, 0.0]) for b in beliefs)


def test_belief_indexer_maps_to_nearest_center():
    centers = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    indexer = BeliefIndexer(centers)

    assert indexer(np.array([0.99, 0.01])) == 0
    assert indexer(np.array([0.49, 0.51])) == 2


def test_policy_optimizer_returns_greedy_belief_policy():
    q_table = np.array([
        [1.0, 2.0, 0.5],
        [3.0, 1.0, 0.5],
        [0.5, 0.5, 2.0],
    ])
    optimizer = PolicyOptimizer(q_table, BeliefIndexer(corner_beliefs(3)), num_actions=3)

    policy = optimizer.greedy_policy()

    assert isinstance(policy, BeliefPolicy)
    np.testing.assert_allclose(policy.policy_table[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(policy.policy_table[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(policy.policy_table[2], [0.0, 0.0, 1.0])


def test_policy_optimizer_softmax_rows_are_distributions():
    q_table = np.array([[1.0, 2.0], [0.5, 0.5]])
    optimizer = PolicyOptimizer(q_table, BeliefIndexer(corner_beliefs(2)), num_actions=2)

    policy = optimizer.softmax_policy(beta=1.0)

    np.testing.assert_allclose(policy.policy_table.sum(axis=1), np.ones(2))
    assert np.all(policy.policy_table >= 0)
