from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..world_model import WorldModel
from ...outcome import Outcome

# Representation of hypothesis parameters of Newcomb world model

@dataclass
class NewcombWorldModelParameters:
    """
    Parameters of Newcomb world model: predictor accuracy
        accuracy = 1.0  predict exactly according to agent's policy
        accuracy = 0.5  predict at random (independent of policy)

    A mixture accuracies:
        coefficients[i]  Mixing coefficient of i-th component
        log_accuracy[i]  [log(1-acc),log(acc)] under i-th component
    """
    coefficients : np.ndarray  # shape (num_components,)
    log_accuracy : np.ndarray  # shape (num_components,2)


@dataclass
class NewcombWorldModelBeliefState:
    """
    Belief state of Newcomb world model: Histogram of previous observations
        history[0]  Number of times predictor was wrong
        history[1]  Number of times predictor was right
    """
    history : np.ndarray  # integer array shape (2,)


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
        assert self.reward_matrix.shape == (self.num_actions, self.num_actions)

    def make_params(self, predictor_accuracy=1) -> NewcombWorldModelParameters:
        assert 0.5 <= predictor_accuracy <= 1.0
        return NewcombWorldModelParameters(coefficients=np.array([1.]),
            log_accuracy=np.log(np.maximum(np.array([[1-predictor_accuracy,predictor_accuracy]]),1e-300)))

    def mix_params(self, params_list: list, coefficients: np.ndarray):
        assert coefficients.shape == (len(params_list),)
        # List of possible outcome distributions is concatenation of distributions from all components
        log_accuracy = np.concatenate([p.log_accuracy for p in params_list], axis=0)
        # Mixture coefficients are product of old and new mixture coefficients
        mixed_coefficients = np.concatenate([p.coefficients * c for p, c in zip(params_list, coefficients)])
        mixed_coefficients /= mixed_coefficients.sum()  # For numerics
        return NewcombWorldModelParameters(coefficients=mixed_coefficients, log_accuracy=log_accuracy)

    def event_index(self, outcome: Outcome) -> int:
        indices = []
        i = outcome.observation            # predicted action
        for j in range(self.num_actions):  # selected action
            if np.isclose(outcome.reward, self.reward_matrix[i,j]):
                indices.append(i*self.num_actions + j)
        if len(indices) == 0:
            raise RuntimeError(f"Invalid outcome in Newcomb environment: {outcome}")
        if len(indices) > 1:
            raise RuntimeError(f"Ambiguous outcome in Newcomb environment: {outcome}")
        return indices[0]

    def initial_state(self):
        return NewcombWorldModelBeliefState(history=np.zeros(2, dtype=np.int64))

    def update_state(self,
            state : NewcombWorldModelBeliefState,
            outcome : Outcome,
            action : int,
            policy : np.ndarray):
        new_state = NewcombWorldModelBeliefState(state.history.copy())
        if np.isclose(policy[action], 1):  # Update only on pure strategies
            # Updating on mixed strategies would be more complicated, as the
            # amount of information gained depends on the policy, e.g. uniform
            # policy -> no information gain because perfect and random
            # predictor behave identical.
            # Also, even a random predictor will be correct sometimes. This is
            # currently not taken into account.
            new_state.history[int(outcome.observation == action)] += 1
        return new_state

    def is_initial(self, state : NewcombWorldModelBeliefState) -> bool:
        return state.history[0] == state.history[1] == 0

    def compute_likelihood(self, belief_state,
            outcome : Outcome,
            params : NewcombWorldModelParameters,
            action : int,
            policy : np.ndarray) -> float:
        """
        P(outcome | belief_state, params, action) under this hypothesis.
        Returns a scalar in [0, 1].
        """
        event = self.event_index(outcome)

        # i: predicted action
        # j: selected action
        i,j = event // self.num_actions, event % self.num_actions

        # Likelihood is just probability of prediction because the action is fixed
        prediction = self._prediction(belief_state, params, policy)
        return prediction[i]

    def compute_expected_reward(self, belief_state,
            reward_function : np.ndarray,
            params : NewcombWorldModelParameters,
            action : int,
            policy : np.ndarray) -> float:
        """
        E[reward_function(outcome) | belief_state, params, action].
        Returns a scalar.
        """
        prediction = self._prediction(belief_state, params, policy)
        reward_matrix = np.reshape(reward_function, (self.num_actions,self.num_actions))
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
        # num_outcomes = num_actions**2; one outcome per (action predicted, action taken) pair
        matrix = np.empty((self.num_actions, self.num_actions**2))
        for i in range(self.num_actions):
            # For each action, fill only the values from the associated column of the reward matrix
            # Leave everything else NaN, to avoid accidentally using wrong values later on
            matrix_i = np.full((self.num_actions,self.num_actions), float("nan"))
            matrix_i[:,i] = self.reward_matrix[:,i]
            matrix[i] = np.reshape(matrix_i, (self.num_actions**2,))

        # Normalise rewards to [0,1]
        assert self.reward_matrix.max() > self.reward_matrix.min()
        return (matrix - self.reward_matrix.min()) / (self.reward_matrix.max() - self.reward_matrix.min())

    def to_str(self, params):
        return "[Newcomb]"

    def _prediction(self,
            state : NewcombWorldModelBeliefState,
            params : NewcombWorldModelParameters,
            policy : np.ndarray) -> np.ndarray:
        """
        Compute/estimate the prediction that the predictor will make
        """

        # Estimate predictor accuracy based hypotheses and observations
        log_likelihood = (params.log_accuracy @ state.history)
        log_likelihood -= log_likelihood.max()  # For numerical stability
        probs = params.coefficients @ np.exp(np.expand_dims(log_likelihood,axis=1) + params.log_accuracy)
        probs /= probs.sum()
        acc = np.clip(probs[1], 0.5, 1.0)

        # Assemble prediction
        perfect_prediction = policy
        random_prediction = np.ones_like(policy) / len(policy)
        prediction = perfect_prediction * (2*acc - 1) \
                    + random_prediction * (2 - 2*acc)
        return prediction
