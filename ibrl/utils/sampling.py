import numpy as np
from numpy.typing import NDArray


def sample_action(rng : np.random.Generator, probabilities: NDArray[np.float64]) -> int:
    """
    Sample an action from a given probability distribution

    Arguments:
        probabilities: Probability distribution over actions

    Returns:
        index of action
    """
    # the distribution should already be normalised, except for possible numerical errors
    assert 0.99 < probabilities.sum() < 1.01
    probabilities = probabilities / probabilities.sum()
    return rng.choice(len(probabilities), p=probabilities)
