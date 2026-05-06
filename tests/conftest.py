"""Shared pytest fixtures."""
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fixed_numpy_seed():
    """Fix numpy global seed before each test for reproducibility.

    Most tests construct their own np.random.default_rng explicitly, but a
    few helper computations (matplotlib defaults, scipy internals) may
    pull from the global RNG. This keeps those deterministic.
    """
    np.random.seed(0)
    yield
