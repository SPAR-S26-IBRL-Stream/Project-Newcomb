import numpy as np
import math
import matplotlib.pyplot as plt

from ibrl.simulators import simulate
from ibrl.utils import construct_environment,dump_array,sample_action
from ibrl.environments import BernoulliBanditEnvironment
from ibrl.agents import InfraBayesianAgent, BernoulliBayesianAgent
from ibrl.infrabayesian.beliefs import BernoulliBelief, NewcombLikeBelief
from ibrl.infrabayesian.a_measure import AMeasure
from ibrl.infrabayesian.infradistribution import Infradistribution
from ibrl.outcome import Outcome


reward_probabilities = [0.5]
n = len(reward_probabilities)
options = {
    "num_actions": n,
    "num_steps":   101,
    "num_runs":    1,
    "seed":        42,
    "verbose":     2,
}
shared = dict(num_actions=n, seed=options["seed"] + 0x01234567, verbose=options["verbose"], epsilon=0.1)


def make_bernoulli(alpha, beta):
    """Create a BernoulliBelief with custom Beta prior."""
    b = BernoulliBelief(num_actions=n)
    b.alpha = np.array(alpha, dtype=float)
    b.beta = np.array(beta, dtype=float)
    return b

agent = InfraBayesianAgent(**shared, beliefs=[
    make_bernoulli([1]*n, [3]*n),  # pessimistic: Beta(1,3), mean=0.25
    make_bernoulli([3]*n, [1]*n),  # optimistic:  Beta(3,1), mean=0.75
])
env = BernoulliBanditEnvironment(probs=reward_probabilities, **options)
#results = simulate(env,agent,options)


env.reset()
agent.reset()
print("Initial agent state:", agent.dump_state())
for step in range(options["num_steps"]):
    probabilities = agent.get_probabilities()
    action = sample_action(agent.random, probabilities)
    outcome = env.step(probabilities, action)
    #outcome.reward = float(step & 1)  # alternate rewards
    agent.update(probabilities, action, outcome)

    print(f"Step:{step:5d}; A={action:2d}; R={outcome.reward:6.2f}; P(A)={dump_array(probabilities)}; Agent state: {agent.dump_state()}")
