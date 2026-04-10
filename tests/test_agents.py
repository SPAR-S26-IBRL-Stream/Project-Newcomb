import numpy as np
import pytest
from ibrl.agents import QLearningAgent, BayesianAgent, EXP3Agent
from ibrl.outcome import Outcome



class TestQLearningAgent:
    def test_initialization(self, num_actions, seed):
        agent = QLearningAgent(num_actions=num_actions, seed=seed)
        assert agent.num_actions == num_actions
        assert agent.seed == seed

    def test_reset(self, q_learning_agent):
        q_learning_agent.reset()
        assert q_learning_agent.step == 1
        assert q_learning_agent.q.shape == (q_learning_agent.num_actions,)
        assert np.allclose(q_learning_agent.q, 0)

    def test_get_probabilities(self, q_learning_agent):
        probs = q_learning_agent.get_probabilities()
        assert probs.shape == (q_learning_agent.num_actions,)
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)

    def test_update(self, q_learning_agent):
        probs = q_learning_agent.get_probabilities()
        action = 0
        reward = 1.0
        outcome = Outcome(reward=reward, env_action=None)
        q_learning_agent.update(probs, action, outcome)
        assert q_learning_agent.step == 2
        assert q_learning_agent.q[action] > 0

    def test_sample_average_increments_action_count(self, num_actions, seed):
        """Verify Q-learning agent increments action count in sample average mode."""
        agent = QLearningAgent(num_actions=num_actions, learning_rate=None, seed=seed)
        agent.reset()
        assert hasattr(agent, 'counts')
        
        probs = agent.get_probabilities()
        outcome = Outcome(reward=1.0, env_action=None)
        agent.update(probs, 0, outcome)
        assert agent.counts[0] == 1


class TestBayesianAgent:
    def test_initialization(self, num_actions, seed):
        agent = BayesianAgent(num_actions=num_actions, seed=seed)
        assert agent.num_actions == num_actions

    def test_reset(self, bayesian_agent):
        bayesian_agent.reset()
        assert bayesian_agent.values.shape == (bayesian_agent.num_actions,)
        assert bayesian_agent.precision.shape == (bayesian_agent.num_actions,)
        assert np.allclose(bayesian_agent.values, 0)

    def test_get_probabilities(self, bayesian_agent):
        probs = bayesian_agent.get_probabilities()
        assert probs.shape == (bayesian_agent.num_actions,)
        assert np.isclose(probs.sum(), 1.0)

    def test_update_increases_precision(self, bayesian_agent):
        initial_precision = bayesian_agent.precision.copy()
        probs = bayesian_agent.get_probabilities()
        outcome = Outcome(reward=1.0, env_action=None)
        bayesian_agent.update(probs, 0, outcome)
        assert bayesian_agent.precision[0] > initial_precision[0]


class TestEXP3Agent:
    def test_initialization(self, num_actions, seed):
        agent = EXP3Agent(num_actions=num_actions, seed=seed)
        assert agent.num_actions == num_actions
        assert agent.gamma == 0.1

    def test_reset(self, exp3_agent):
        exp3_agent.reset()
        assert exp3_agent.log_weights.shape == (exp3_agent.num_actions,)
        assert np.allclose(exp3_agent.log_weights, 0)

    def test_get_probabilities(self, exp3_agent):
        probs = exp3_agent.get_probabilities()
        assert probs.shape == (exp3_agent.num_actions,)
        assert np.isclose(probs.sum(), 1.0)
        assert np.all(probs >= 0)

    def test_update_changes_weights(self, exp3_agent):
        initial_weights = exp3_agent.log_weights.copy()
        probs = exp3_agent.get_probabilities()
        outcome = Outcome(reward=1.0, env_action=None)
        exp3_agent.update(probs, 0, outcome)
        assert not np.allclose(exp3_agent.log_weights, initial_weights)
        
        
