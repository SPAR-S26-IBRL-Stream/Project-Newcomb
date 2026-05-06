"""Tests for IntervalKernelBelief."""
import numpy as np
import pytest

from ibrl.mdp.interval_belief import IntervalKernelBelief


def _make_initial_polytope(num_states=3, num_actions=2):
    """Toy polytope: cell 0 ∈ [0.5, 0.7], cell 1 ∈ [0.2, 0.4],
    cell 2 ∈ [0.0, 0.1] for every (s, a)."""
    p_lo = np.zeros((num_states, num_actions, num_states))
    p_hi = np.zeros((num_states, num_actions, num_states))
    p_lo[..., 0] = 0.5; p_hi[..., 0] = 0.7
    p_lo[..., 1] = 0.2; p_hi[..., 1] = 0.4
    p_lo[..., 2] = 0.0; p_hi[..., 2] = 0.1
    return p_lo, p_hi


class TestPosteriorMean:
    def test_initial_mean_is_uniform_over_active_cells(self):
        b = IntervalKernelBelief(num_states=3, num_actions=2, alpha_init=0.5)
        m = b.posterior_mean()
        np.testing.assert_allclose(m, np.full((3, 2, 3), 1.0 / 3))

    def test_initial_mean_excludes_masked_cells(self):
        # If the initial polytope says some cells are impossible, prior α=0 there
        p_lo, p_hi = _make_initial_polytope()
        # Make cell 2 impossible at all (s, a)
        p_hi[..., 2] = 0.0
        b = IntervalKernelBelief(num_states=3, num_actions=2,
                                 alpha_init=0.5, initial_polytope=(p_lo, p_hi))
        m = b.posterior_mean()
        np.testing.assert_allclose(m[..., 2], 0.0)
        # Active cells split mass 0.5/0.5
        np.testing.assert_allclose(m[..., 0], 0.5)
        np.testing.assert_allclose(m[..., 1], 0.5)

    def test_update_concentrates_on_observed_successor(self):
        b = IntervalKernelBelief(num_states=3, num_actions=2, alpha_init=0.5)
        for _ in range(20):
            b.update(0, 0, 1)
        m = b.posterior_mean()
        # Cell 1 should dominate after 20 obs
        assert m[0, 0, 1] > 0.7


class TestPolytopeShrinkage:
    def test_polytope_at_unvisited_matches_initial(self):
        p_lo_init, p_hi_init = _make_initial_polytope()
        b = IntervalKernelBelief(num_states=3, num_actions=2,
                                 alpha_init=0.5,
                                 initial_polytope=(p_lo_init, p_hi_init))
        plo, phi = b.polytope(confidence=0.95)
        # Unvisited (s=1, a=0) should fall back to the initial polytope
        # because Beta credible interval intersect with prior is too sparse
        # to override.
        assert plo[1, 0, 0] == pytest.approx(p_lo_init[1, 0, 0])
        assert phi[1, 0, 0] == pytest.approx(p_hi_init[1, 0, 0])

    def test_polytope_shrinks_with_observations_inside_initial(self):
        p_lo_init, p_hi_init = _make_initial_polytope()
        b = IntervalKernelBelief(num_states=3, num_actions=2,
                                 alpha_init=0.5,
                                 initial_polytope=(p_lo_init, p_hi_init))
        # Many observations consistent with the initial polytope center
        # (cell 0 hit ~60% of the time, cells 1/2 split the rest)
        for _ in range(60):
            b.update(0, 0, 0)
        for _ in range(30):
            b.update(0, 0, 1)
        for _ in range(10):
            b.update(0, 0, 2)
        plo, phi = b.polytope(confidence=0.95)
        initial_width = (p_hi_init - p_lo_init)[0, 0]
        final_width = (phi - plo)[0, 0]
        # Widths should NOT have grown; cell 0 in particular should shrink
        # (its empirical mean ≈ 0.6, near the polytope center)
        assert (final_width <= initial_width + 1e-9).all()
        assert final_width[0] < initial_width[0]

    def test_polytope_remains_simplex_feasible(self):
        # Even with sparse contradictory data, sum_lo <= 1 <= sum_hi at every (s, a)
        p_lo_init, p_hi_init = _make_initial_polytope()
        b = IntervalKernelBelief(num_states=3, num_actions=2,
                                 alpha_init=0.5,
                                 initial_polytope=(p_lo_init, p_hi_init))
        for _ in range(2):
            b.update(0, 0, 2)  # rare cell
        plo, phi = b.polytope(confidence=0.95)
        sum_lo = plo.sum(axis=2)
        sum_hi = phi.sum(axis=2)
        assert (sum_lo <= 1.0 + 1e-9).all()
        assert (sum_hi >= 1.0 - 1e-9).all()
