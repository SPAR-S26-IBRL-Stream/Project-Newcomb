import numpy as np
import pytest

from ibrl.infrabayesian import NewcombWorldModel
from ibrl.infrabayesian.a_measure import AMeasure
from ibrl.infrabayesian.infradistribution import Infradistribution
from ibrl.outcome import Outcome


def newcomb_prediction(policy: np.ndarray, accuracy: float) -> np.ndarray:
    """Analytic predictor distribution used by Newcomb-like environments."""
    perfect_prediction = policy
    random_prediction = np.ones_like(policy) / len(policy)
    return (
        perfect_prediction * (2 * accuracy - 1)
        + random_prediction * (2 - 2 * accuracy)
    )


def test_newcomb_likelihood_matches_predictor_boundary_cases():
    """Perfect and random predictors have the expected event likelihoods."""
    wm = NewcombWorldModel()
    state = wm.initial_state()
    committed_policy = np.array([0.25, 0.75])

    perfect_params = wm.make_params(predictor_accuracy=1.0)
    random_params = wm.make_params(predictor_accuracy=0.5)

    # Perfect predictors copy the committed policy.
    for observed_action, expected_likelihood in enumerate(committed_policy):
        outcome = Outcome(
            observation=observed_action,
            reward=wm.reward_matrix[observed_action, 1],
        )
        np.testing.assert_array_equal(
            wm.compute_likelihood(
                state,
                outcome,
                perfect_params,
                1,
                committed_policy,
            ),
            expected_likelihood,
        )

    # Random predictors ignore the committed policy.
    for observed_action, expected_likelihood in enumerate(np.array([0.5, 0.5])):
        outcome = Outcome(
            observation=observed_action,
            reward=wm.reward_matrix[observed_action, 1],
        )
        np.testing.assert_array_equal(
            wm.compute_likelihood(
                state,
                outcome,
                random_params,
                1,
                committed_policy,
            ),
            expected_likelihood,
        )


def test_newcomb_expected_reward_matches_prediction_table_product():
    """compute_expected_reward should implement prediction @ reward_table[:, action]."""
    reward_table = np.array([[10., 15.], [0., 5.]])
    wm = NewcombWorldModel(reward_table)
    state = wm.initial_state()
    params = wm.make_params(predictor_accuracy=0.7)
    policy = np.array([0.4, 0.6])

    normalized_table = (
        reward_table - reward_table.min()
    ) / (reward_table.max() - reward_table.min())
    prediction = newcomb_prediction(policy, accuracy=0.7)

    for action in [0, 1]:
        expected = prediction @ normalized_table[:, action]
        actual = wm.compute_expected_reward(
            state,
            wm.agent_reward_matrix()[action],
            params,
            action,
            policy,
        )
        np.testing.assert_array_equal(actual, expected)


def test_newcomb_updates_history_only_for_pure_policies():
    """Newcomb belief history records predictor correctness only for pure policies."""
    wm = NewcombWorldModel()
    state = wm.initial_state()

    state = wm.update_state(
        state,
        Outcome(observation=0, reward=wm.reward_matrix[0, 0]),
        action=0,
        policy=np.array([1., 0.]),
    )
    np.testing.assert_array_equal(state.history, np.array([0, 1]))

    state = wm.update_state(
        state,
        Outcome(observation=1, reward=wm.reward_matrix[1, 0]),
        action=0,
        policy=np.array([1., 0.]),
    )
    np.testing.assert_array_equal(state.history, np.array([1, 1]))

    state = wm.update_state(
        state,
        Outcome(observation=0, reward=wm.reward_matrix[0, 0]),
        action=0,
        policy=np.array([0.5, 0.5]),
    )
    np.testing.assert_array_equal(state.history, np.array([1, 1]))


@pytest.mark.parametrize(
    "reward_table,accuracy",
    [
        (np.array([[10., 11.], [0., 1.]]), 0.9),  # Newcomb, boxA=1, boxB=10
        (np.array([[0., 10.], [10., 0.]]), 0.8),  # Death in Damascus
        (np.array([[0., 10.], [10., 5.]]), 0.7),  # Asymmetric Death in Damascus
        (np.array([[2., 0.], [0., 1.]]), 0.9),    # Coordination game
    ],
)
def test_newcomb_agent_policy_maximizes_discretized_table_value(
    reward_table: np.ndarray,
    accuracy: float,
):
    """Convert the notebook's validation plot into a deterministic policy test."""
    from ibrl.agents import InfraBayesianAgent

    wm = NewcombWorldModel(reward_table)
    dist = Infradistribution([AMeasure(wm.make_params(accuracy))], wm)
    agent = InfraBayesianAgent(
        num_actions=2,
        epsilon=0.,
        hypotheses=[dist],
        reward_function=wm.agent_reward_matrix(),
        policy_discretisation=5,
    )
    agent.reset()

    expected_values = np.array([
        newcomb_prediction(policy, accuracy) @ reward_table @ policy
        for policy in agent.policies
    ])
    optimal = np.isclose(expected_values, expected_values.max())
    expected_policy = (
        agent.policies * np.expand_dims(optimal, axis=1)
    ).sum(axis=0) / optimal.sum()

    np.testing.assert_allclose(agent.get_probabilities(), expected_policy)
