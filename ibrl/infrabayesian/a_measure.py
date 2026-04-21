import numpy as np


class AMeasure:
    """
    An a-measure, characterised by
        a scale factor λ > 0
        a probability measure μ (encoded as opaque params — WorldModel defines the structure)
        an offset b >= 0

    AMeasure is a pure data container. All model-family logic (how to compute
    probabilities, likelihoods, and expected rewards from params) lives in WorldModel.
    Belief state (the sufficient statistic of observations) lives in Infradistribution.
    """
    def __init__(self, params, scale : float = 1, offset : float = 0):
        assert offset >= 0
        assert scale > 0
        self.params = params
        self.scale = np.float64(scale)
        self.offset = np.float64(offset)

    def evaluate_action(self, world_model, belief_state, reward_function : np.ndarray,
                        action : int, policy=None) -> float:
        """
        Compute the scaled and shifted expected value of a given reward function, defined as λ*μ(f) + b
        """
        return self.scale * world_model.compute_expected_reward(
            belief_state, reward_function, self.params, action=action, policy=policy
        ) + self.offset

    def reset(self):
        """
        Reset internal state
        """
        self.scale = np.float64(1)
        self.offset = np.float64(0)

    # Helper functions

    def __itruediv__(self, other : float):
        """
        In-place division by a scalar
        """
        self.scale /= other
        self.offset /= other
        return self

    def __repr__(self) -> str:
        return f"({self.scale:.3f}[...],{self.offset:.3f})"
