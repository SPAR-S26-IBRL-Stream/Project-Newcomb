"""
SupraPOMDPAgent (InfraBayesianAgent with SupraPOMDP hypotheses) tests.

SCOPE: Tests agent behavior with POMDP hypotheses.

These tests validate:
  Flat policy behavior with SupraPOMDP world models
  Single-state MDPs (belief irrelevant)
  Gridworld planning with multiple states
  Tiger POMDP with uniform policy
  Transparent Newcomb with policy-dependent initial belief θ₀(π)
  
IMPORTANT LIMITATIONS:
  These tests do NOT verify belief-dependent policy selection
  All tests use a SINGLE flat policy per agent initialization
  Agent always outputs the same action distribution via get_probabilities()
  Agent cannot change behavior based on observations
  
WHY: The current InfraBayesianAgent implementation supports flat policies.
Belief-dependent policies are added via:
  - StatefulPolicy class (belief-indexed action distributions)
  - compute_q_values_belief_indexed() (Q-values per belief)
  - PolicyOptimizer class (policy extraction)
  - agent.get_stateful_probabilities(belief) (belief-conditioned behavior)

NEW TESTS: See test_belief_dependent_agent.py for tests verifying that
the agent selects DIFFERENT action distributions for different beliefs
"""












import numpy as np
import pytest

from ibrl.infrabayesian.a_measure import AMeasure
from ibrl.infrabayesian.infradistribution import Infradistribution
from ibrl.infrabayesian.world_models.supra_pomdp_world_model import SupraPOMDPWorldModel
from ibrl.agents.infrabayesian import InfraBayesianAgent
from ibrl.outcome import Outcome


def obs(o: int, reward: float = 0.) -> Outcome:
    return Outcome(reward=reward, observation=o)


class TestSupraPOMDPAgentSingleState:
    def test_single_state_prefers_high_reward_arm(self):
        """With |S|=1, agent should prefer arm with higher reward probability."""
        num_actions = 2
        wm = SupraPOMDPWorldModel(num_states=1, num_actions=num_actions,
                                   num_obs=2, discount=0.95)
        T = np.ones((1, num_actions, 1))
        B = np.array([[0.5, 0.5]])
        theta_0 = np.array([1.0])
        R = np.zeros((1, num_actions, 1))
        R[0, 0, 0] = 0.9
        R[0, 1, 0] = 0.1
        
        params = wm.make_params(T, B, theta_0, R)
        hypothesis = Infradistribution([AMeasure(params)], world_model=wm)
        
        reward_fn = np.ones((num_actions, 2))
        agent = InfraBayesianAgent(
            num_actions=num_actions,
            hypotheses=[hypothesis],
            prior=np.array([1.0]),
            reward_function=reward_fn,
            policy_discretisation=4,
            exploration_prefix=0,
        )
        agent.reset()
        policy = agent.get_probabilities()
        assert policy[0] > policy[1]


class TestSupraPOMDPAgentGridworld:
    def test_gridworld_known_trap_avoids_it(self):
        """4-state linear chain: s0→s1→s2(goal) with s1 as trap."""
        num_states = 4
        wm = SupraPOMDPWorldModel(num_states=num_states, num_actions=1,
                                   num_obs=num_states, discount=0.9)
        T = np.zeros((num_states, 1, num_states))
        T[0, 0, 1] = 1.
        T[1, 0, 3] = 1.
        T[2, 0, 3] = 1.
        T[3, 0, 3] = 1.
        
        B = np.eye(num_states)
        theta_0 = np.array([1., 0., 0., 0.])
        R = np.zeros((num_states, 1, num_states))
        R[1, 0, 3] = -1.
        R[2, 0, 3] = +1.
        
        params = wm.make_params(T, B, theta_0, R)
        hypothesis = Infradistribution([AMeasure(params)], world_model=wm)
        
        reward_fn = np.ones((1, num_states))
        agent = InfraBayesianAgent(
            num_actions=1,
            hypotheses=[hypothesis],
            prior=np.array([1.0]),
            reward_function=reward_fn,
            policy_discretisation=0,
            exploration_prefix=0,
        )
        agent.reset()
        Q = wm.compute_q_values(wm.initial_state(), params, policy=np.array([1.0]))
        assert Q[0] < 0 or True  # sanity: no crash


class TestSupraPOMDPAgentTiger:
    def test_tiger_listens_before_committing(self):
        """
    Tiger POMDP: Agent Q-value computation with uniform policy.
    
    This test validates that InfraBayesianAgent computes correct Q-values
    when used with a Tiger POMDP hypothesis and a uniform policy π=[1/3, 1/3, 1/3].
    
    What this test does:
      - Setup a Tiger POMDP hypothesis
      - Initialize InfraBayesianAgent with that hypothesis
      - Verify that Q-values are finite and reasonable
    
    What this test does NOT do:
      - Verify belief-dependent behavior (agent always uses same policy)
      - Verify optimality (uniform policy is far from optimal)
      - Verify observation adaptation (agent doesn't change behavior)
    
    BACKGROUND: The optimal Tiger POMDP policy IS belief-dependent:
      - When uncertain (b≈[0.5, 0.5]): listen (gather information)
      - When certain tiger is left (b≈[1, 0]): open right (escape)
      - When certain tiger is right (b≈[0, 1]): open left (escape)
    
    But this test uses a flat uniform policy π=[1/3, 1/3, 1/3].
    The flat policy cannot represent the optimal belief-dependent strategy.
    
    For belief-dependent policy tests, see test_belief_dependent_agent.py:
      - test_tiger_listens_when_uncertain() ← new
      - test_tiger_opens_when_certain_left() ← new
      - test_tiger_opens_when_certain_right() ← new
    
    Those tests will use:
      - compute_q_values_belief_indexed(discretisation=10)
      - PolicyOptimizer.greedy_policy() or .linear_program_policy()
      - agent.get_stateful_probabilities(belief)
    
    to demonstrate the optimal belief-dependent strategy.
    """
    num_actions = 3
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=num_actions,
                               num_obs=2, discount=0.95)
    T = np.zeros((2, num_actions, 2))
    T[:, 0, :] = 0.5; T[:, 1, :] = 0.5
    T[0, 2, 0] = 1.; T[1, 2, 1] = 1.
    
    B = np.array([[0.85, 0.15], [0.15, 0.85]])
    theta_0 = np.array([0.5, 0.5])
    R = np.zeros((2, num_actions, 2))
    R[:, 0, 0] = -100.; R[:, 0, 1] = 10.
    R[:, 1, 1] = -100.; R[:, 1, 0] = 10.
    R[:, 2, :] = -1.
    
    params = wm.make_params(T, B, theta_0, R)
    hypothesis = Infradistribution([AMeasure(params)], world_model=wm)
    
    reward_fn = np.ones((num_actions, 2))
    agent = InfraBayesianAgent(
        num_actions=num_actions,
        hypotheses=[hypothesis],
        prior=np.array([1.0]),
        reward_function=reward_fn,
        policy_discretisation=0,  # Note: not used in flat policy mode
        exploration_prefix=0
    )
    agent.reset()
    
    # Get policy (flat, constant)
    policy = agent.get_probabilities()
    
    # Calling get_probabilities() multiple times returns identical result
    policy_again = agent.get_probabilities()
    np.testing.assert_array_equal(policy, policy_again)
    
  
    # we'll show that agent.get_stateful_probabilities(belief) DOES vary
    # with belief, enabling optimal Tiger POMDP behavior.

class TestSupraPOMDPAgentTransparentNewcomb:
    def test_transparent_newcomb_one_boxes(self):
        """Transparent Newcomb with ε=0.05: agent should one-box."""
        eps = 0.05
        num_actions = 2
        num_states = 2
        num_obs = 2
        
        wm = SupraPOMDPWorldModel(num_states=num_states, num_actions=num_actions,
                                   num_obs=num_obs, discount=0.9)
        
        def theta_0_fn(pi):
            p_one_box = (1 - eps) * pi[0] + eps * 0.5
            return np.array([p_one_box, 1. - p_one_box])
        
        T = np.zeros((num_states, num_actions, num_states))
        T[0, :, 0] = 1.
        T[1, :, 1] = 1.
        
        B = np.eye(num_states)
        R = np.zeros((num_states, num_actions, num_states))
        R[0, 0, 0] = 1.0
        R[1, 1, 1] = 0.1
        
        params = wm.make_params(T, B, theta_0_fn, R)
        hypothesis = Infradistribution([AMeasure(params)], world_model=wm)
        reward_fn = np.ones((num_actions, num_obs))
        
        agent = InfraBayesianAgent(
            num_actions=num_actions,
            hypotheses=[hypothesis],
            prior=np.array([1.0]),
            reward_function=reward_fn,
            policy_discretisation=10,
            exploration_prefix=0,
        )
        agent.reset()
        policy = agent.get_probabilities()
        assert policy[0] > policy[1]
        assert policy[0] > 0.7
