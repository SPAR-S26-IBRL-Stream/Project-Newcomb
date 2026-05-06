"""End-to-end smoke tests for the four MDP agents on a small gridworld."""
import numpy as np
import pytest

from ibrl.mdp import (
    GridworldEnvironment, IntervalKernelBelief, simulate_mdp_multi_episode,
    RobustDPAgent, BayesianDPAgent, ThompsonDPAgent, IBDPAgent,
)


def _setup(eps=0.05, rows=3, cols=3):
    env = GridworldEnvironment(rows=rows, cols=cols, epsilon=eps, seed=0,
                               reward_pos=(rows - 1, cols - 1),
                               trap_pos=(0, cols - 1))
    env.reset()
    P_lo, P_hi = env.kernel_polytope()
    return env, P_lo, P_hi


def _make_agent(cls, env, polytope, seed=0):
    base = dict(num_states=env.num_states, num_actions=env.num_actions,
                R=env.R, terminal_mask=env.terminal_mask, gamma=env.gamma,
                seed=seed)
    if cls is RobustDPAgent:
        return cls(**base, initial_polytope=polytope)
    belief = IntervalKernelBelief(num_states=env.num_states,
                                  num_actions=env.num_actions,
                                  alpha_init=0.5, initial_polytope=polytope)
    return cls(**base, belief=belief)


class TestEndToEndOnSmallGrid:
    @pytest.mark.parametrize("agent_cls",
                             [RobustDPAgent, BayesianDPAgent,
                              ThompsonDPAgent, IBDPAgent])
    def test_runs_without_error(self, agent_cls):
        env, P_lo, P_hi = _setup()
        agent = _make_agent(agent_cls, env, (P_lo, P_hi))
        r = simulate_mdp_multi_episode(env, agent,
                                       num_episodes=3, max_steps_per_episode=30,
                                       seed=42)
        assert r["episode_returns"].shape == (3,)
        assert r["cumulative_regret"].shape == (3,)


class TestRealizableConvergence:
    @pytest.mark.parametrize("agent_cls",
                             [RobustDPAgent, BayesianDPAgent, IBDPAgent])
    def test_eps_zero_low_late_regret(self, agent_cls):
        # With ε=0 and a clear path to reward, agents should reach near-zero
        # late-episode regret. Thompson is excluded because PSRL with sparse
        # Dirichlet posterior can be very noisy on small samples.
        env_template = GridworldEnvironment(rows=4, cols=4, epsilon=0.0, seed=0,
                                            reward_pos=(3, 3), trap_pos=(1, 2))
        env_template.reset()
        P_lo, P_hi = env_template.kernel_polytope()
        env = GridworldEnvironment(rows=4, cols=4, epsilon=0.0, seed=0,
                                   reward_pos=(3, 3), trap_pos=(1, 2))
        agent = _make_agent(agent_cls, env, (P_lo, P_hi))
        r = simulate_mdp_multi_episode(env, agent, num_episodes=5,
                                       max_steps_per_episode=50, seed=1)
        # Late episodes should have small absolute regret (allow up to 0.3
        # given the env's stochasticity and the discrete return)
        assert abs(r["episode_regret"][-2:].mean()) < 0.3


class TestRobustDPDoesNotReplan:
    def test_policy_is_constant_across_episodes(self):
        env, P_lo, P_hi = _setup()
        agent = _make_agent(RobustDPAgent, env, (P_lo, P_hi))
        agent.reset()
        agent.plan()
        policy_ep0 = agent._policy.copy()
        agent.reset_episode()
        agent.plan()
        policy_ep1 = agent._policy
        np.testing.assert_array_equal(policy_ep0, policy_ep1)


class TestIBDPPolytopeShrinks:
    def test_polytope_width_does_not_grow(self):
        env, P_lo, P_hi = _setup()
        agent = _make_agent(IBDPAgent, env, (P_lo, P_hi))
        plo0, phi0 = agent.belief.polytope(confidence=0.95)
        simulate_mdp_multi_episode(env, agent, num_episodes=3,
                                   max_steps_per_episode=30, seed=42)
        plo1, phi1 = agent.belief.polytope(confidence=0.95)
        # Width should not increase at any cell
        width_before = (phi0 - plo0)
        width_after = (phi1 - plo1)
        assert (width_after <= width_before + 1e-9).all()
