# IB agent architecture proposal

This is an explanation of the updated IB agent architecture.

## Motivation
Some issues with the current implementation are explained in this document (discussed in the group meeting on April 6th):

https://docs.google.com/document/d/1uWw0lb2lXMwLBy7XyCC5d7pFaN-qolcE6VKVaF5Urek/edit?tab=t.0

In particular, it motivates that the IB agent architecture should be:

- A classical prior (probability distribution)
- Over hypotheses (infradistributions), each of which describes one or more
- Possible worlds/Probability distributions over observations (a-measures)

Here, a possible world is a complete description of the world (potentially probabilistic), e.g. "the coin has a 60% chance of landing heads". A possible world is encoded as an a-measure.

A hypothesis represents Knightian Uncertainty over possible worlds, e.g. "I do not know whether the chance of the coin landing heads is 60% or 80%". A hypothesis is encoded as an infradistribution, whose minimal points are the a-measures. An infradistribution with a single minimal point represents absence of KU and should reproduce classical behaviour.

A prior represents classical uncertainty between hypotheses, e.g. "I think that is a 60% chance that the coin lands heads with 50% probability and a 40% chance that the coin lands heads with 75% probability". Just like in a Bayesian agent, this is represented by a probability distribution over hypotheses.


## Design considerations
Conceptually, an a-measure assigns a probability to all possible histories. Even when truncating histories, it is computationally infeasible to represent a-measures like this for any remotely realistic scenario. Another approach is needed to encode them in the code.

Infradistributions are infinite sets of sa-measures. However, all of their behaviour is encoded in their minimal points, which is an infinite set of a-measures. Even more, we only need to consider the extremal minimal points. These are the points that are necessary to span the convex hull of the minimal points. This is a finite and usually not too large set of a-measures, so we expect that it is possible to store them explicitly.

A mixture over infradistributions is itself an infradistribution. In particular, a classical distribution over infradistribution is also an infradistribution. We can therefore merge the first and second layer of the architecture stack described above and only maintain a single mixed infradistribution throughout. A very useful side effect of this convention is that we do not need to update the classical distribution explicitly (which would be somewhat unclear, since it would require the likelihood over an infradistribution). For a mixed infradistribution we only need the IB update rule to update everything.

## Implementation
The following discusses the architecture in the context a Bernoulli bandit, i.e. the agent chooses one of several actions and each action has a fixed probability of receiving a reward of either 0 or 1. Importantly, the outcomes are discrete. The implementation could easily be generalised to more discrete outcomes but not yet to continuous rewards.

### Discrete Bayesian agent
Before we turn to the IB agent, we consider a naive Bayesian agent. This agent makes the key design concepts more visible and helps to understand the IB generalisation. This agent is implemented in `ibrl/agents/discrete_bayesian.py`.

The agent operates in an environment with discrete outcomes, i.e. instead of a real-valued reward, the agent receives an integer, representing a discrete observation. It has a utility function (`self.reward_function`) that assigns to each outcome a real-valued reward in `[0,1]`.

The agent further has an array `self.hypotheses` of shape `(num_hypotheses,num_outcomes)`. The entry `hypotheses(h,o)` is the probability of outcome `o` happening under hypothesis `h` and `Σ_o hypotheses(h,o) = 1` for all `h`.

The agent maintains a set of distributions over hypotheses, `self.prior` of shape `(num_actions,num_hypotheses)`. For each action, this represents the probability that the agent assigns to each hypothesis. These distributions are initialised uniformly. As the agent learns, only this distribution gets updated.

When making a decision, the agent uses the prior and the hypotheses to determine expected rewards of each action and then takes the action that maximises it.

### Infrabayesian agent
The IB agent differs from the discrete Bayesian in the following ways:

- instead of a prior distribution over hypothesis it maintains a mixed infradistribution for each action; instead of a classical update, we use the IB update rule
- expected rewards are computed using the infra-expected values; the hypotheses array is only used to initialise the infradistributions

Learning happens via updates to the infradistribution.

**TODO: describe this more**


### A-measures
Conceptually, an a-measure `(λμ,b)` consists of a scale `λ>0`, an offset `b≥0` and a probability measure `μ : X → [0,1]`. The space `X` is the set of all possible histories of some fixed length `N`, i.e. `X = O^N`.

We can construct the probability measure from a reward probability. Say the probability of getting a reward is `p`. For a history in which we got a reward `A` times and did not get a reward `B` times, the probability of that history is `p^A (1-p)^B`. Applying this to all histories of length `N`, we can construct the full measure.

We can mix a-measure by taking linear combinations `c1 (λ1 μ1,b1) + c2 (λ2 μ2,b2) = (c1 λ1 μ1 + c2 λ2 μ2, b1 + b2)`. Note that a mixed a-measure no longer corresponds to a single reward probability `p`. By mixing we lose information about which probabilities and mixing coefficients were used. In particular, we lose the ability to extend histories later on. Therefore we need to pick `N` large enough to encompass the entire lifetime of the agent.

As described, the computational complexity of this procedure scales exponentially with the history length. The actual code parametrises measures in a such way that components can be constructed on-the-fly in constant time. Effectively, we only write out those histories that actually matter. See `ibrl/infrabayesian/a_measure.py` for implementation details. Conceptually, this approach is equivalent to mapping out all the histories for some very large `N`, as described above. 

**TODO: describe IB update rule** The a-measure implementation changes the form of the update rule a bit

**TODO: implement mixture properly**: Right now, the code creates mixed a-measures and initialises a single infradistribution from them. Conceptually, it should create multiple infradistributions and then mix them (at least when reproducing the classical agent). We might want both functionalities in the end.


See `experiments/fllor2/ibtest.ipynb` for validation that the IB agent, when initialised similar to the classical agent, exactly reproduces the classical agent.

