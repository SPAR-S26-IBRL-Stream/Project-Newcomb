import pytest
import numpy as np
from ibrl.agents import QLearningAgent, BayesianAgent, EXP3Agent
from ibrl.environments import (
    BanditEnvironment, NewcombEnvironment, DeathInDamascusEnvironment,
    CoordinationGameEnvironment, PolicyDependentBanditEnvironment,
    AsymmetricDeathInDamascusEnvironment, SwitchingAdversaryEnvironment,
    MatchEnvironment, ReverseTailsEnvironment
)


@pytest.fixture
def seed():
    return 42

@pytest.fixture
def num_actions():
    return 2

@pytest.fixture
def num_steps():
    return 10


# Agents - NO parametrization here (keep individual fixtures)
@pytest.fixture
def q_learning_agent(num_actions, seed):
    agent = QLearningAgent(num_actions=num_actions, seed=seed)
    agent.reset()
    return agent

@pytest.fixture
def bayesian_agent(num_actions, seed):
    agent = BayesianAgent(num_actions=num_actions, seed=seed)
    agent.reset()
    return agent

@pytest.fixture
def exp3_agent(num_actions, seed):
    agent = EXP3Agent(num_actions=num_actions, seed=seed)
    agent.reset()
    return agent


# Environments - NO parametrization here (keep individual fixtures)
@pytest.fixture
def bandit_env(num_actions, seed):
    env = BanditEnvironment(num_actions=num_actions, seed=seed)
    env.reset()
    return env

@pytest.fixture
def newcomb_env(seed):
    env = NewcombEnvironment(num_actions=2, seed=seed)
    env.reset()
    return env

@pytest.fixture
def damascus_env(seed):
    env = DeathInDamascusEnvironment(num_actions=2, seed=seed)
    env.reset()
    return env

@pytest.fixture
def coordination_env(seed):
    env = CoordinationGameEnvironment(num_actions=2, seed=seed)
    env.reset()
    return env

@pytest.fixture
def pdbandit_env(seed):
    env = PolicyDependentBanditEnvironment(num_actions=2, seed=seed)
    env.reset()
    return env

@pytest.fixture
def asymmetric_damascus_env(seed):
    env = AsymmetricDeathInDamascusEnvironment(num_actions=2, seed=seed)
    env.reset()
    return env

@pytest.fixture
def switching_env(seed):
    env = SwitchingAdversaryEnvironment(num_actions=2, num_steps=100, seed=seed)
    env.reset()
    return env

@pytest.fixture
def match_env(seed):
    env = MatchEnvironment(num_actions=2, seed=seed)
    env.reset()
    return env

@pytest.fixture
def reverse_tails_env(seed):
    env = ReverseTailsEnvironment(num_actions=2, seed=seed)
    env.reset()
    return env


@pytest.fixture
def agent_and_env(q_learning_agent, bandit_env):
    return q_learning_agent, bandit_env

@pytest.fixture
def random_state(seed):
    return np.random.default_rng(seed=seed)


def pytest_configure(config):
    markers = [
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
        "property: marks tests as property-based tests",
        "convergence: marks tests as convergence tests",
        "adversarial: marks tests as adversarial/edge case tests",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)
