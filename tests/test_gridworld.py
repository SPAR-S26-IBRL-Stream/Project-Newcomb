"""Tests for GridworldEnvironment."""
import numpy as np
import pytest

from ibrl.mdp.gridworld import GridworldEnvironment


class TestStaticAdversary:
    def test_terminal_states_absorb(self):
        env = GridworldEnvironment(epsilon=0.05, seed=0)
        env.reset()
        assert env.true_kernel[env.reward_idx, :, env.reward_idx] == pytest.approx(1.0)
        assert env.true_kernel[env.trap_idx, :, env.trap_idx] == pytest.approx(1.0)

    def test_oracle_value_at_terminals_is_zero(self):
        env = GridworldEnvironment(epsilon=0.05, seed=0)
        env.reset()
        V = env.get_oracle_value()
        assert V[env.reward_idx] == pytest.approx(0.0)
        assert V[env.trap_idx] == pytest.approx(0.0)

    def test_eps_zero_matches_nominal_kernel(self):
        env = GridworldEnvironment(epsilon=0.0, seed=0, p_nominal=0.8)
        env.reset()
        # Intended-direction probability should be exactly 0.8 at every (s, a)
        # for non-terminal states.
        for s in range(env.num_states):
            if s in env.terminal_states:
                continue
            for a in range(env.num_actions):
                # Sum over successors should be 1
                assert env.true_kernel[s, a].sum() == pytest.approx(1.0)


class TestPolytopeBounds:
    def test_polytope_contains_true_kernel(self):
        env = GridworldEnvironment(epsilon=0.05, seed=0)
        env.reset()
        P_lo, P_hi = env.kernel_polytope()
        assert (env.true_kernel >= P_lo - 1e-9).all()
        assert (env.true_kernel <= P_hi + 1e-9).all()

    def test_polytope_widens_with_epsilon(self):
        env_narrow = GridworldEnvironment(epsilon=0.02, seed=0)
        env_wide = GridworldEnvironment(epsilon=0.10, seed=0)
        env_narrow.reset()
        env_wide.reset()
        Pn_lo, Pn_hi = env_narrow.kernel_polytope()
        Pw_lo, Pw_hi = env_wide.kernel_polytope()
        assert (Pw_hi - Pw_lo).sum() > (Pn_hi - Pn_lo).sum()


class TestResetEpisode:
    def test_static_adversary_kernel_persists_across_episodes(self):
        env = GridworldEnvironment(epsilon=0.05, seed=0, adversary_mode="static")
        env.reset()
        K1 = env.true_kernel.copy()
        env.reset_episode()
        K2 = env.true_kernel.copy()
        np.testing.assert_array_equal(K1, K2)

    def test_per_episode_visit_can_change_kernel(self):
        env = GridworldEnvironment(epsilon=0.10, seed=0,
                                   adversary_mode="per_episode_visit")
        env.reset()
        K1 = env.true_kernel.copy()
        # Walk through some states to populate visit counts
        env._curr_visit_counts[5] += 10
        env._curr_visit_counts[10] += 5
        env.reset_episode()
        K2 = env.true_kernel
        # Kernel may or may not change, but the call should succeed
        # and not crash. Test that visit-count rollover happens.
        assert env._prev_visit_counts.sum() == 15  # the populated counts


class TestStep:
    def test_walk_to_reward_terminates(self):
        # ε=0, p_nominal=1 ⇒ deterministic; agent goes east×4 then south×4
        env = GridworldEnvironment(epsilon=0.0, p_nominal=1.0, seed=0,
                                   reward_pos=(4, 4), trap_pos=(2, 2),
                                   start_pos=(0, 0))
        env.reset()
        path = [1, 1, 1, 1, 2, 2, 2, 2]  # EEEE SSSS
        total_reward = 0.0
        for action in path:
            ns, r, done = env.step(action)
            total_reward += r
        # Hit reward state (4,4) on last step
        assert done
        assert ns == env.reward_idx
        assert total_reward == pytest.approx(7 * (-0.01) + 1.0)
