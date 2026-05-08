from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import WorldModel
from ...outcome import Outcome


@dataclass
class NewcombWorldModelParameters:
    """
    Parameters of Newcomb world model.

    A mixture of accuracies:
        coefficients[i]  Mixing coefficient of i-th component
        log_accuracy[i]  [log(1-acc), log(acc)] under i-th component
    """
    coefficients: np.ndarray
    log_accuracy: np.ndarray


@dataclass
class NewcombWorldModelBeliefState:
    """
    Belief state of Newcomb world model: histogram of previous observations.

    This compact right/wrong history is only a sufficient statistic for pure
    policies. For mixed policies, the infradistribution update still applies
    the immediate observation likelihood to each a-measure's scale, but this
    persistent predictor-accuracy history is left unchanged.
    """
    history: np.ndarray


class NewcombWorldModel(WorldModel):
    """World model for Newcomb-like predictor accuracy hypotheses."""

    def __init__(self, reward_matrix=None):
        # reward_matrix[i, j] is reward upon prediction i and action j.
        if reward_matrix is None:
            self.reward_matrix = np.array([
                [10, 15],  # prediction 0 (one-box) -> second box filled
                [0, 5],    # prediction 1 (two-box) -> second box empty
            ])
        else:
            self.reward_matrix = np.array(reward_matrix)

        self.num_actions = self.reward_matrix.shape[0]
        assert self.reward_matrix.shape == (self.num_actions, self.num_actions)

    def make_params(self, predictor_accuracy: float = 1.0) -> NewcombWorldModelParameters:
        assert 0.5 <= predictor_accuracy <= 1.0
        accuracy = np.array([[1 - predictor_accuracy, predictor_accuracy]])
        return NewcombWorldModelParameters(
            coefficients=np.array([1.0]),
            log_accuracy=np.log(np.maximum(accuracy, 1e-300)),
        )

    def mix_params(
            self,
            params_list: list[NewcombWorldModelParameters],
            coefficients: np.ndarray) -> NewcombWorldModelParameters:
        assert coefficients.shape == (len(params_list),)
        log_accuracy = np.concatenate([p.log_accuracy for p in params_list], axis=0)
        mixed_coefficients = np.concatenate([
            p.coefficients * c for p, c in zip(params_list, coefficients)
        ])
        mixed_coefficients /= mixed_coefficients.sum()
        return NewcombWorldModelParameters(
            coefficients=mixed_coefficients,
            log_accuracy=log_accuracy,
        )

    def event_index(self, outcome: Outcome, action: int) -> int:
        prediction = outcome.observation
        if prediction is None or not (0 <= prediction < self.num_actions):
            raise RuntimeError(f"Invalid outcome in Newcomb environment: {outcome}")
        if not (0 <= action < self.num_actions):
            raise RuntimeError(f"Invalid action in Newcomb environment: {action}")
        if not np.isclose(outcome.reward, self.reward_matrix[prediction, action]):
            raise RuntimeError(f"Invalid outcome in Newcomb environment: {outcome}")
        return prediction * self.num_actions + action

    def initial_state(self) -> NewcombWorldModelBeliefState:
        return NewcombWorldModelBeliefState(history=np.zeros(2, dtype=np.int64))

    def update_state(
            self,
            state: NewcombWorldModelBeliefState,
            outcome: Outcome,
            action: int,
            policy: np.ndarray | None,
            params=None) -> NewcombWorldModelBeliefState:
        new_state = NewcombWorldModelBeliefState(state.history.copy())
        if policy is not None and np.isclose(policy[action], 1):
            new_state.history[int(outcome.observation == action)] += 1
        return new_state

    def is_initial(self, state: NewcombWorldModelBeliefState) -> bool:
        return state.history[0] == state.history[1] == 0

    def compute_likelihood(
            self,
            belief_state: NewcombWorldModelBeliefState,
            outcome: Outcome,
            params: NewcombWorldModelParameters,
            action: int,
            policy: np.ndarray | None) -> float:
        event = self.event_index(outcome, action)
        prediction = event // self.num_actions
        return float(self._prediction(belief_state, params, policy)[prediction])

    def compute_expected_reward(
            self,
            belief_state: NewcombWorldModelBeliefState,
            reward_function: np.ndarray,
            params: NewcombWorldModelParameters,
            action: int,
            policy: np.ndarray | None) -> float:
        prediction = self._prediction(belief_state, params, policy)
        reward_matrix = np.reshape(reward_function, (self.num_actions, self.num_actions))
        rewards = prediction @ reward_matrix
        return float(rewards[action])

    def agent_reward_matrix(self) -> np.ndarray:
        """
        Convert the world model's reward table to the agent reward_function layout.
        """
        matrix = np.empty((self.num_actions, self.num_actions ** 2))
        for action in range(self.num_actions):
            matrix_i = np.full((self.num_actions, self.num_actions), float("nan"))
            matrix_i[:, action] = self.reward_matrix[:, action]
            matrix[action] = np.reshape(matrix_i, (self.num_actions ** 2,))

        assert self.reward_matrix.max() > self.reward_matrix.min()
        return (matrix - self.reward_matrix.min()) / (
            self.reward_matrix.max() - self.reward_matrix.min()
        )

    def to_str(self, params):
        return "[Newcomb]"

    def _prediction(
            self,
            state: NewcombWorldModelBeliefState,
            params: NewcombWorldModelParameters,
            policy: np.ndarray | None) -> np.ndarray:
        if policy is None:
            policy = np.ones(self.num_actions) / self.num_actions

        log_likelihood = params.log_accuracy @ state.history
        log_likelihood -= log_likelihood.max()
        probs = params.coefficients @ np.exp(
            np.expand_dims(log_likelihood, axis=1) + params.log_accuracy
        )
        probs /= probs.sum()
        acc = np.clip(probs[1], 0.5, 1.0)

        perfect_prediction = policy
        random_prediction = np.ones_like(policy) / len(policy)
        return (
            perfect_prediction * (2 * acc - 1)
            + random_prediction * (2 - 2 * acc)
        )
