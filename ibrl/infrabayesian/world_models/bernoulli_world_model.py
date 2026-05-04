from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..world_model import WorldModel
from ...outcome import Outcome


# Representation of hypothesis parameters and belief state of Bernoulli world model

@dataclass
class BernoulliWorldModelParameters:
    """
    Parameters of Bernoulli world model: For each arm, a mixture probability distributions over outcomes
        coefficients[a][i]  Mixing coefficient of i-th hypothesis for arm a
        log_probs[a][i][j]  Log probability of outcome j under i-th hypothesis for arm a
    """
    coefficients : list[np.ndarray]  # shape of arrays (num_components,)
    log_probs    : list[np.ndarray]  # shape of arrays (num_components,num_outcomes)

@dataclass
class BernoulliWorldModelBeliefState:
    """
    Belief state of Bernoulli world model: Histogram of previous observations
        history[a][i]  Number of times outcome i was observed from arm a
    """
    history : np.ndarray  # integer array shape (num_arms, num_outcomes)



class MultiBernoulliWorldModel(WorldModel):
    """
    World model for a multi-arm Bernoulli bandit.

    Construct params via make_params(arm_distributions) where arm_distributions[a] is
    either a single probability array (shape num_outcomes) or a list of such arrays.
    """

    def __init__(self, num_arms: int, num_outcomes: int = 2):
        self.num_arms = num_arms
        self.num_outcomes = num_outcomes

    def make_params(self, distributions: list[list[np.ndarray] | np.ndarray]) -> BernoulliWorldModelParameters:
        """
        Create parameter object from probability distribution

        Arguments:
            arm_hypothesis: one of
                    list (for each arm) of lists of probability distributions of each outcome
                    list (for each arm) of probability distributions of each outcome
                If multiple distributions are specified, a uniform mixture is created.
                For more complex mixtures, see mix_params below.
        """
        assert len(distributions) == self.num_arms
        params = BernoulliWorldModelParameters([], [])
        for arm_distributions in distributions:
            # Accept list[array] as well as list[list[array]]
            if isinstance(arm_distributions, np.ndarray):
                arm_distributions = [arm_distributions]

            # Outcome probabilities. Shape: (num_components,num_outcomes)
            probs = np.stack(arm_distributions)
            assert np.allclose(probs.sum(axis=1), 1), "Outcome probabilities must sum to 1"
            params.log_probs.append(np.log(np.maximum(probs, 1e-300)))  # Lower cut-off to avoid log(0)

            # Uniform mixture
            params.coefficients.append(np.ones(len(arm_distributions))/len(arm_distributions))
        return params

    def mix_params(self, params_list: list[BernoulliWorldModelParameters], coefficients: np.ndarray) -> BernoulliWorldModelParameters:
        """Mix per-arm params independently across arms."""
        mixed = BernoulliWorldModelParameters([],[])
        for arm in range(self.num_arms):
            # List of possible outcome distributions is concatenation of distributions from all components
            log_probs = np.concatenate([p.log_probs[arm] for p in params_list], axis=0)
            # Mixture coefficients are product of old and new mixture coefficients
            mixed_coefficients = np.concatenate([p.coefficients[arm] * c for p, c in zip(params_list, coefficients)])
            mixed_coefficients /= mixed_coefficients.sum()  # For numerics
            mixed.log_probs.append(log_probs)
            mixed.coefficients.append(mixed_coefficients)
        return mixed

    def event_index(self, outcome: Outcome, action : int) -> int:
        # For Bernoulli bandits reward is the event type, so this mapping is exact.
        return int(round(outcome.reward * (self.num_outcomes - 1)))

    def initial_state(self) -> BernoulliWorldModelBeliefState:
        return BernoulliWorldModelBeliefState(np.zeros((self.num_arms, self.num_outcomes), dtype=np.int64))

    def update_state(self,
            state: BernoulliWorldModelBeliefState,
            outcome: Outcome,
            action: int,
            policy: np.ndarray | None) -> BernoulliWorldModelBeliefState:
        new_state = BernoulliWorldModelBeliefState(state.history.copy())
        new_state.history[action, self.event_index(outcome, action)] += 1
        return new_state

    def is_initial(self, state: BernoulliWorldModelBeliefState) -> bool:
        return not state.history.any()

    def compute_likelihood(self,
            belief_state: BernoulliWorldModelBeliefState,
            outcome: Outcome,
            params : BernoulliWorldModelParameters,
            action: int,
            policy: np.ndarray | None) -> float:
        probs = self._predictive(belief_state.history[action], params.log_probs[action], params.coefficients[action])
        return float(probs[self.event_index(outcome, action)])

    def compute_expected_reward(self,
            belief_state: BernoulliWorldModelBeliefState,
            reward_function: np.ndarray,
            params : BernoulliWorldModelParameters,
            action: int,
            policy: np.ndarray | None) -> float:
        probs = self._predictive(belief_state.history[action], params.log_probs[action], params.coefficients[action])
        return float(probs @ reward_function)

    def to_str(self, params : BernoulliWorldModelParameters) -> str:
        """
        String representation of measure:
            [meas1;meas2;...]
        where measX is the measure for arm X:
            comp1,comp2,...
        where compY is the Y-th component of the measure:
            c:{p1,p2,...}
        where c is the mixing coefficient and pZ is the probability of the Z-th outcome

        E.g.: 1 arm, 2 components (60-40 mix of p=0.9 and p=0.8), 2 outcomes (1-p,p):
            -> [0.60:{0.1,0.9},0.40:{0.2,0.8}]
        """
        per_arm_params = []
        for arm in range(self.num_arms):
            components = []
            for c,p in zip(params.coefficients[arm],np.exp(params.log_probs[arm])):
                components.append(f"""{c:.2f}:{{{",".join(f"{pp:.1f}" for pp in p)}}}""")
            per_arm_params.append(",".join(components))
        return "[" + ";".join(per_arm_params) + "]"

    def _predictive(self,
            arm_counts : np.ndarray,
            log_probs : np.ndarray,
            coefficients : np.ndarray) -> np.ndarray:
        """Posterior predictive P(next outcome | arm history) for mixture of categoricals."""
        lp = (log_probs * arm_counts).sum(axis=1)
        lp -= lp.max()  # shift for numerical stability before exp
        probs = coefficients @ np.exp(np.expand_dims(lp, axis=1) + log_probs)
        return probs / probs.sum()
