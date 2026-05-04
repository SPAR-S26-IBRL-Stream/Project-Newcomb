import numpy as np
import pytest
from ibrl.infrabayesian.world_models.supra_pomdp_world_model import SupraPOMDPWorldModel
from ibrl.infrabayesian.a_measure import AMeasure
from ibrl.infrabayesian.infradistribution import Infradistribution
from ibrl.agents.infrabayesian import InfraBayesianAgent
from ibrl.outcome import Outcome


def obs(o: int, reward: float = 0.) -> Outcome:
    return Outcome(reward=reward, observation=o)


class TestBeliefDependentPolicyTiger:
    """
    Tiger POMDP: Classic POMDP where optimal policy is belief-dependent.
    
    States: 0 = tiger left, 1 = tiger right
    Actions: 0 = listen, 1 = open left, 2 = open right
    Observations: 0 = growl left, 1 = growl right
    
    Optimal behavior: listen until confident, then open opposite door from growl.
    """
    
    @pytest.fixture
    def tiger_environment(self):
        num_states = 2
        num_actions = 3
        num_obs = 2
        
        wm = SupraPOMDPWorldModel(num_states=num_states, num_actions=num_actions,
                                   num_obs=num_obs, discount=0.95)
        
        # Transitions
        T = np.zeros((num_states, num_actions, num_states))
        T[:, 0, :] = 0.5  # Listen: uniform next state
        T[:, 1, :] = 0.5  # Open left: uniform next state
        T[0, 2, 0] = 1.0  # Open right from state 0: stay in state 0
        T[1, 2, 1] = 1.0  # Open right from state 1: stay in state 1
        
        # Observation model
        B = np.array([
            [0.85, 0.15],  # State 0 (tiger left): hear left 85%, right 15%
            [0.15, 0.85]   # State 1 (tiger right): hear left 15%, right 85%
        ])
        
        # Initial belief
        theta_0 = np.array([0.5, 0.5])
        
        # Rewards
        R = np.zeros((num_states, num_actions, num_states))
        R[:, 0, :] = -1.0  # Listening costs 1
        R[0, 1, 1] = -100.0  # Open left when tiger is left: bad
        R[0, 1, 0] = 10.0   # Open left when tiger is right: good
        R[1, 2, 1] = -100.0  # Open right when tiger is right: bad
        R[1, 2, 0] = 10.0   # Open right when tiger is left: good
        
        params = wm.make_params(T, B, theta_0, R)
        return wm, params
    
    def test_belief_dependent_policy_listens_when_uncertain(self, tiger_environment):
        """Agent should listen when uncertain about tiger location."""
        wm, params = tiger_environment
        
        hypothesis = Infradistribution([AMeasure(params)], world_model=wm)
        
        reward_fn = np.ones((3, 2))  # Simple: reward all observations
        agent = InfraBayesianAgent(
            num_actions=3,
            hypotheses=[hypothesis],
            prior=np.array([1.0]),
            reward_function=reward_fn,
            policy_discretisation=4,
            policy_optimization="greedy",
            exploration_prefix=0
        )
        agent.reset()
        
        # At uniform belief (uncertain), agent should prefer listening
        uncertain_belief = np.array([0.5, 0.5])
        policy_uncertain = agent.get_stateful_probabilities(uncertain_belief)
        
        # Listening (action 0) should have highest probability
        assert policy_uncertain[0] >= policy_uncertain[1]
        assert policy_uncertain[0] >= policy_uncertain[2]
    
    def test_belief_dependent_policy_opens_when_certain(self, tiger_environment):
        """Agent should open door when certain about tiger location."""
        wm, params = tiger_environment
        
        hypothesis = Infradistribution([AMeasure(params)], world_model=wm)
        reward_fn = np.ones((3, 2))
        
        agent = InfraBayesianAgent(
            num_actions=3,
            hypotheses=[hypothesis],
            prior=np.array([1.0]),
            reward_function=reward_fn,
            policy_discretisation=4,
            policy_optimization="greedy",
            exploration_prefix=0
        )
        agent.reset()
        
        # Belief certain tiger is left
        certain_left = np.array([1.0, 0.0])
        policy_left = agent.get_stateful_probabilities(certain_left)
        
        # Should prefer opening right (action 2) to escape tiger
        assert policy_left[2] > policy_left[0]
        
        # Belief certain tiger is right
        certain_right = np.array([0.0, 1.0])
        policy_right = agent.get_stateful_probabilities(certain_right)
        
        # Should prefer opening left (action 1) to escape tiger
        assert policy_right[1] > policy_right[0]
