import numpy as np


def sample_action(rng : np.random.Generator, probabilities: np.ndarray) -> int:
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
