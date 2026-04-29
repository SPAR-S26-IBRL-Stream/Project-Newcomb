from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .base import WorldModel
from ...outcome import Outcome

# Representation of hypothesis parameters of Newcomb world model

@dataclass
class NewcombWorldModelParameters:
    """
    Parameters of Newcomb world model: predictor accuracy
        predictor_accuracy=1.0  predict exactly according to agent's policy
        predictor_accuracy=0.5  predict at random (independent of policy)
    """
    predictor_accuracy : float


# Outcomes: {0, A, B, A+B} with small box A, large box B
# Actions: {1, 2} for one/two box
# Params: predictor accuracy
class NewcombWorldModel(WorldModel):
    """
    Defines the belief state type, update logic, likelihood computation,
    and hypothesis param construction for a model family. Stateless —
    belief state is owned by Infradistribution.

    Belief state and params are opaque to callers — structure is defined
    by each subclass and documented there.
    """

    def __init__(self, reward_matrix = None):
        # Convention:
        # reward = prediction @ reward_matrix @ policy
        #        = Σ_ij prediction[i] reward_matrix[i,j] policy[j]
        # i.e. reward_matrix[i,j] is reward upon prediction i and action j
        if reward_matrix is None:
            # Reward matrix of Newcomb problem
            self.reward_matrix = np.array([
                [10, 15],  # prediction 0 (one-box) -> second box filled
                [ 0,  5]   # prediction 1 (two-box) -> second box empty
            ])
        else:
            self.reward_matrix = reward_matrix
        
        self.num_actions = self.reward_matrix.shape[0]

    def make_params(self, predictor_accuracy=1) -> NewcombWorldModelParameters:
        assert 0.5 <= predictor_accuracy <= 1.0
        return NewcombWorldModelParameters(float(predictor_accuracy))

    def mix_params(self, params_list: list, coefficients: np.ndarray):
        # TODO: Not yet sure how to mix
        assert len(params_list) == 1
        return params_list[0]

    def event_index(self, outcome: Outcome, action : int) -> int:
        for i in range(self.num_actions):
            for j in range(self.num_actions):
                if np.isclose(outcome.reward, self.reward_matrix[i,j]) and j==action:
                    return i*self.num_actions + j
        raise RuntimeError(f"Invalid outcome in Newcomb environment: {outcome}")

    def initial_state(self):
        return None

    def update_state(self, state,
            outcome : Outcome,
            action : int,
            policy : np.ndarray,
            params=None):
        pass

    def is_initial(self, state) -> bool:
        return True

    def compute_likelihood(self, belief_state,
            outcome : Outcome,
            params : NewcombWorldModel,
            action : int,
            policy : np.ndarray) -> float:
        """
        P(outcome | belief_state, params, action) under this hypothesis.
        Returns a scalar in [0, 1].
        """
        event = self.event_index(outcome, action)

        # i: predicted action
        # j: selected action
        i,j = event // self.num_actions, event % self.num_actions

        perfect_prediction = policy
        random_prediction = np.ones_like(policy) / len(policy)
        prediction = perfect_prediction * (2*params.predictor_accuracy - 1) \
                    + random_prediction * (2 - 2*params.predictor_accuracy)

        # Likelihood is just probability of prediction because the action is fixed (?)
        return prediction[i]
        #return prediction[i]*policy[j]

    def compute_expected_reward(self, belief_state,
            reward_function : np.ndarray,
            params : NewcombWorldModel,
            action : int,
            policy : np.ndarray) -> float:
        """
        E[reward_function(outcome) | belief_state, params, action].
        Returns a scalar.
        """
        perfect_prediction = policy
        random_prediction = np.ones_like(policy) / len(policy)
        prediction = perfect_prediction * (2*params.predictor_accuracy - 1) \
                    + random_prediction * (2 - 2*params.predictor_accuracy)
        #rewards = prediction @ self.reward_matrix
        reward_matrix = np.array([
            [reward_function[0], reward_function[1]],
            [reward_function[2], reward_function[3]]
        ])
        rewards = prediction @ reward_matrix
        return rewards[action]

    def agent_reward_matrix(self) -> np.ndarray:
        """
        The world model stores the reward matrix as a (num_actions,num_action) array,
            each element indicating the reward for a given (action predicted, action taken) pair
        The agent stores the reward matrix as a (num_actions,num_outcomes) array,
            each element indicating the reward of a given outcome after taking a given action
        This function converts the world model's reward table to a format suitable for the agent
        """
        matrix = np.array([
            [self.reward_matrix[0,0], float("nan"), self.reward_matrix[1,0], float("nan")],  # action one-box
            [float("nan"), self.reward_matrix[0,1], float("nan"), self.reward_matrix[1,1]]   # action two-box
        ])
        # Normalise rewards to [0,1]
        return (matrix - self.reward_matrix.min()) / (self.reward_matrix.max() - self.reward_matrix.min())

    def to_str(self, params):
        return "[Newcomb]"
