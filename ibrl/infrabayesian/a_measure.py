import numpy as np


class AMeasure:
    """
    An a-measure (λμ, b): scale λ > 0, probability measure μ encoded as
    hypothesis params (opaque — WorldModel defines the structure), offset b ≥ 0.
    Stateless with respect to observations — belief state lives in Infradistribution.
    """
    def __init__(self, params, scale: float = 1.0, offset: float = 0.0):
        assert offset >= 0
        assert scale > 0
        self.params = params
        self.scale = np.float64(scale)
        self.offset = np.float64(offset)

    def __itruediv__(self, other: float):
        self.scale /= other
        self.offset /= other
        return self

    def reset(self):
        self.scale = np.float64(1)
        self.offset = np.float64(0)

    def evaluate_action(self, world_model, belief_state, reward_function: np.ndarray,
                        action: int, policy=None) -> float:
        """Compute λ·E_μ[reward_function] + b for this a-measure."""
        raw = world_model.compute_expected_reward(
            belief_state, reward_function, self.params, action=action, policy=policy
        )
        return self.scale * raw + self.offset

    def __repr__(self) -> str:
        return f"(λ={self.scale:.3f}, b={self.offset:.3f})"
