# Infrabayesian Reinforcement Learning Experiment

## Description
This directory contains the infrastructure to run many different environment/agent/policy combinations. Plots are generated to illustrate obtained rewards.

## Instructions
To run this experiment, execute
```bash
make -f experiments/agent_comparison_sweep/Makefile [-j N]
```
from the project root directory. Ideally use multiple cores, as the runs take a while to complete.

For creating plots, `gnuplot` must be installed.

## Environments
The implementation focuses on bandit-like environments, i.e. environments consisting of a single time-step and where no information is available to the agent, prior to making its decision. Restricting to these environments avoids much of the complexity usually associated with this problem.

The following environments are currently implemented:[^1]
- **Classical multi-armed bandit**: A fixed number of discrete action are available to the agent. A reward is sampled from a different probability distribution, depending on the action.
- **Newcomb's problem**: The agent chooses to take only box B or boxes A and B. Box A is always filled with a small reward. Box B is filled with a large reward, but only if the agent is predicted not to take box A.
- **Death in Damascus**: The agent chooses to go to either the city of Damascus or Aleppo. Death knows the agent's policy and also goes to one of the cities. If they end up in the same city, the agent dies.
- **Asymmetric Death in Damascus**: As above, but the agent prefers to die in Aleppo, rather than Damascus.
- **Coordination game**: The agent plays a cooperative game against another agent with identical policy.
- **Policy-dependent bandit**: A generalisation of both the classical multi-armed bandit and of Newcomb-like environments. A reward is sampled from a different probability distribution for each (prediction,action) pair. The probability distributions are determined randomly.
- **Switching bandit**: Like the multi-armed bandit, but the reward moves to a different arm.

## Agents
The following agents are investigated:
1) Classical **Q-learning agent**:[^2] The agent uses either an epsilon-greedy or a softmax policy (with decaying epsilon/temperature) to encourage exploration.
2) **Bayesian agent**: Similar to Q-learning, but maintains a probability distribution over possible rewards. At each update, the central value and uncertainty get updated based on the new observation.
3) **EXP3 agent**: Implementation of the "Exponential-weight algorithm for Exploration and Exploitation".
4) **Experimental agent 1**: Similar to Q-learning, but instead of returning a non-deterministic policy, it samples an action from that policy and then returns a deterministic policy that chooses this action. The predictor will thus deterministically predict that action. We therefore only access the diagonal entries of the reward table and are able to learn them via classical methods. If this optimal policy is deterministic, this agent is expected to converge to it.
5) **Experimental agent 2**: This agent aims to learn the entire reward table, as that allows constructing the optimal policy explicitly, even if it is non-deterministic. The agent achieves this by picking strongly peaked (but still non-deterministic) policies, such that it can be fairly confident in what the predictor chose. Since we know the action taken and action predicted, we can then update the corresponding entry of the reward table. Whenever we pick an action that is different from the most likely one, we can update the off-diagonal entries of the table. Learning off-diagonal elements is slow, but we eventually reconstruct the entire reward table.
6) **Experimental agent 3**: In an MDP, we try to find the optimal (discrete) action. In an NDP, we try to find the optimal (continuous) probability distribution. If we discretise the probability distributions, we can effectively turn an NDP into a (much larger) MDP. This connection was noticed by Emanuel. He also noted that it would be a bad idea to turn this into an agent.


## Results
For each environment 2000 independent runs are performed. At each time step, the average reward is calculated. All tests (except the EXP3 agent) use a policy that strongly encourages exploration at the beginning with then linear decreases down to a minimum after 500 steps. The behaviour of an agent should thus only be judged after step 500.

### Classical agents
| Q-learning agent                                   | Bayesian agent                                     | EXP3 agent                                         |
| -------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- |
| ![](figures/bandit.classical.png)                  | ![](figures/bandit.bayesian.png)                   | ![](figures/bandit.exp3.png)                       |
| ![](figures/newcomb.classical.png)                 | ![](figures/newcomb.bayesian.png)                  | ![](figures/newcomb.exp3.png)                      |
| ![](figures/damascus.classical.png)                | ![](figures/damascus.bayesian.png)                 | ![](figures/damascus.exp3.png)                     |
| ![](figures/asymmetric-damascus.classical.png)     | ![](figures/asymmetric-damascus.bayesian.png)      | ![](figures/asymmetric-damascus.exp3.png)          |
| ![](figures/coordination.classical.png)            | ![](figures/coordination.bayesian.png)             | ![](figures/coordination.exp3.png)                 |
| ![](figures/pdbandit.classical.png)                | ![](figures/pdbandit.bayesian.png)                 | ![](figures/pdbandit.exp3.png)                     |
| ![](figures/switching.classical.png)               | ![](figures/switching.bayesian.png)                | ![](figures/switching.exp3.png)                    |

We find that the classical agents converge close to the optimal policy on the multi-armed bandit environment, but fail to do so in the Newcomb-like environments. An interesting exception is the Bayesian agent in the Death in Damascus environment. Note that the spread of individual runs is quite large. There are runs in which the classical agent achieves close-to-optimal reward on Newcomb-like environments. But even then, it does not converge on the optimal policy and looking at the plots we clearly see that we can not rely on the agent to behave sensibly on average.

### Experimental agents
The experimental agents are designed to gain insights into Newcomb-like problems. These results do not necessarily reflect infrabayesianism.

| Experimental agent 1                               | Experimental agent 2                               | Experimental agent 3                               |
| -------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- |
| ![](figures/bandit.experimental1.png)              | N/A                                                | N/A                                                |
| ![](figures/newcomb.experimental1.png)             | ![](figures/newcomb.experimental2.png)             | ![](figures/newcomb.experimental3.png)             |
| ![](figures/damascus.experimental1.png)            | ![](figures/damascus.experimental2.png)            | ![](figures/damascus.experimental3.png)            |
| ![](figures/asymmetric-damascus.experimental1.png) | ![](figures/asymmetric-damascus.experimental2.png) | ![](figures/asymmetric-damascus.experimental3.png) |
| ![](figures/coordination.experimental1.png)        | ![](figures/coordination.experimental2.png)        | ![](figures/coordination.experimental3.png)        |
| ![](figures/pdbandit.experimental1.png)            | ![](figures/pdbandit.experimental2.png)            | ![](figures/pdbandit.experimental3.png)            |
| ![](figures/switching.experimental1.png)           | ![](figures/switching.experimental2.png)           | ![](figures/switching.experimental3.png)           |


The experimental agent 1 is able to converge on the best deterministic policy. In Newcomb's problem and the coordination game, these are the optimal policies. In Death in Damascus, this necessarily yields reward 0. In Newcomb's problem, some runs do not converge to the optimal policy within 1000 steps. This because a fast cool down of the exploration parameter was chosen. With a sufficiently slow cool down or given sufficient time, all runs will converge to the optimal policy.

For technical reasons, experimental agents 2 and 3 cannot yet operate in the bandit environment. In all other environment, it is able to converge on a reasonably optimal policy. These agents tend to converge slowly. These plots are only possible after enforcing a lot of exploration. The policy-dependent bandit seems to be especially difficult for the agents. Even with more time and exploration, they struggle to converge to an optimal policy.


[^1]: *Reinforcement Learning in Newcomblike Environments*, Proceedings to NeurIPS 2021, [PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/b9ed18a301c9f3d183938c451fa183df-Paper.pdf)
[^2]: *Reinforcement Learning: An Introduction*, Richard S. Sutton and Andrew G. Barto, Second Edition, MIT Press, Cambridge, MA, 2018, [PDF](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
