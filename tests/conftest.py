import pytest
import numpy as np
from ibrl.agents import QLearningAgent, BayesianAgent, EXP3Agent
from ibrl.environments import (
    BanditEnvironment, NewcombEnvironment, DeathInDamascusEnvironment,
    CoordinationGameEnvironment, PolicyDependentBanditEnvironment
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
