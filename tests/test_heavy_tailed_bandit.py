"""Sanity checks for HeavyTailedBanditEnvironment."""
import numpy as np
import pytest

from ibrl.environments.heavy_tailed_bandit import HeavyTailedBanditEnvironment


class TestArmStructure:
    def test_arm_parameters_persist_across_reset_episode(self):
        env = HeavyTailedBanditEnvironment(num_actions=4, seed=42)
        env.reset()
        medians_before = env.medians.copy()
        env.reset_episode()
        np.testing.assert_array_equal(env.medians, medians_before,
                                      "medians must persist across reset_episode")

    def test_arm_parameters_reshuffle_on_full_reset(self):
        env = HeavyTailedBanditEnvironment(num_actions=4, seed=42)
        env.reset()
        m1 = env.medians.copy()
        env.reset()
        m2 = env.medians.copy()
        # different RNG draws → different arms (almost surely)
        assert not np.allclose(m1, m2)

    def test_optimal_reward_equals_best_median(self):
        env = HeavyTailedBanditEnvironment(num_actions=5, seed=42)
        env.reset()
        assert env.get_optimal_reward() == env.medians.max()


class TestHeavyTailedDraws:
    def test_median_concentrates_to_arm_median(self):
        env = HeavyTailedBanditEnvironment(num_actions=3, seed=0)
        env.reset()
        rewards = np.array([env.interact(0) for _ in range(20_000)])
        # median (not mean — Cauchy tails make sample mean unstable)
        assert abs(np.median(rewards) - env.medians[0]) < 0.2, (
            f"empirical median {np.median(rewards):.3f} should approximate "
            f"arm median {env.medians[0]:.3f}")

    def test_sample_variance_is_huge_due_to_cauchy_contamination(self):
        env = HeavyTailedBanditEnvironment(
            num_actions=3, seed=0, contam_low=0.10, contam_high=0.10,
            gamma_low=3.0, gamma_high=3.0)
        env.reset()
        rewards = np.array([env.interact(0) for _ in range(5_000)])
        # Pure Gaussian noise would give variance ~1; Cauchy contamination
        # (10 percent prob, gamma=3) blows the sample variance up. The exact
        # number is RNG-dependent so we only assert it's well above 1.
        assert rewards.var() > 5.0, (
            f"Cauchy contamination should yield large sample variance, got {rewards.var():.2f}")


class TestContaminationBounds:
    def test_invalid_bounds_rejected(self):
        with pytest.raises(AssertionError):
            HeavyTailedBanditEnvironment(num_actions=2, contam_low=0.5, contam_high=0.2)
        with pytest.raises(AssertionError):
            HeavyTailedBanditEnvironment(num_actions=2, gamma_low=-1.0, gamma_high=2.0)
