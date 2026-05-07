# IB agent architecture proposal

This is an explanation of the updated IB agent architecture.

## Motivation
Some issues with a prior implementation motivate the architecture below.
In particular, the IB agent architecture should be:

- A classical prior (probability distribution)
- Over hypotheses (infradistributions), each of which describes one or more
- Possible worlds/Probability distributions over observations (a-measures)

Here, a possible world is a complete description of the world (potentially probabilistic), e.g. "the coin has a 60% chance of landing heads". A possible world is encoded as an a-measure.

A hypothesis represents Knightian Uncertainty over possible worlds, e.g. "I do not know whether the chance of the coin landing heads is 60% or 80%". A hypothesis is encoded as an infradistribution, whose minimal points are the a-measures. An infradistribution with a single minimal point represents absence of KU and reproduces classical behaviour.

A prior represents classical uncertainty between hypotheses, e.g. "I think that is a 60% chance that [the coin lands heads with 50% probability] and a 40% chance that [the coin lands heads with 75% probability]". Just like in a Bayesian agent, this is represented by a probability distribution over hypotheses.


## Design considerations
Conceptually, an a-measure assigns a probability to all possible histories. Even when truncating histories, it is computationally infeasible to represent a-measures like this for any realistic scenario. Another approach is needed to encode them practically.

Infradistributions are infinite sets of sa-measures. However, all of their behaviour is encoded in their minimal points, which is an infinite set of a-measures. Even more, we only need to consider the extremal minimal points. These are the points that are necessary to span the convex hull of the minimal points. This is a finite and usually not too large set of a-measures, so we expect that it is possible to store them explicitly.

A mixture over infradistributions is itself an infradistribution. In particular, a classical distribution over infradistributions is also an infradistribution. We can therefore merge the first and second layer of the architecture stack described above and only maintain a single mixed infradistribution throughout. A very useful side effect of this convention is that we do not need to update the classical distribution explicitly (which would be somewhat unclear, since it would require the likelihood over an infradistribution). For a mixed infradistribution we only need the IB update rule to update everything.

## Implementation
The following discusses the architecture in the context a Bernoulli bandit, i.e. the agent chooses one of several actions and each action has a fixed probability of receiving a reward of either 0 or 1. Importantly, the outcomes are discrete. The implementation could easily be generalised to more discrete outcomes but not yet to continuous rewards.

### Discrete Bayesian agent
Before we turn to the IB agent, we consider a naive Bayesian agent. This agent makes the key design concepts more visible and helps to understand the IB generalisation. This agent is implemented in `ibrl/agents/discrete_bayesian.py`.

1. The agent operates in an environment with discrete outcomes, i.e. instead of a real-valued reward, the agent receives an integer, representing a discrete observation. It has a utility function that assigns to each outcome a reward in $[0,1]$.
2. The agent further has an array `hypotheses` of shape `(num_hypotheses,num_outcomes)`. The entry `hypotheses[h,o]` is the probability of outcome `o` happening under hypothesis `h` and `Σ_o hypotheses[h,o] = 1` for all `h`.
3. The agent maintains a set of distributions over hypotheses, `prior`, of shape `(num_actions,num_hypotheses)`. For each action, this represents the probability that the agent assigns to each hypothesis. These distributions are initialised uniformly. As the agent learns, only this distribution gets updated.
4. When making a decision, the agent uses the prior and the hypotheses to determine expected rewards of each action and then takes the action that maximises it. It uses an ε-greedy policy to encourage exploration.

### Infrabayesian agent
### Classical mode of operation
When operating without KU, the IB agent differs from the discrete Bayesian in the following ways:

1. The agent operates on the same discrete outcomes as the classical agent. It uses the same concept of a reward function.
2. The agent has the same array `hypotheses` that the classical agent uses, though it has a slightly different meaning. For the classical agent, this array actually represents the hypotheses of the agent throughout its operation. It gets used during the update and to compute expectation values. For the IB agent, the `hypotheses` array is used only to initialise the infradistribution(s), which are now the objects that represent hypotheses. Updates and expectation values are computed only from the infradistributions and do not need this array. The individual infradistributions, corresponding to each row of `hypotheses`, are then mixed according to a uniform prior.
3. The mixed infradistribution replaces the classical `prior`. Just like the classical agent had one prior for each action, the IB agent has one infradistribution for each action. These infradistributions are completely independent of each other. Learning happens by updating the infradistribution of the selected action. The IB update rule replaces the classical update rule.
4. When making a decision, the agent computes the infra-expected value of each action and takes the action that maximises it. Like the classical agent, it uses an ε-greedy policy to encourage exploration.

It was validated that the IB agent exactly reproduces the classical agent, when initialised in this way. See `experiments/ib_architecture/ib_validate_classical.ipynb` for details.

Before turning to the true IB behaviour, it is helpful to explain the implementation of the mathematical objects.


### A-measures
Conceptually, an a-measure $(λμ,b)$ consists of a scale $λ>0$, an offset $b≥0$ and a probability measure $μ : X → [0,1]$. The space $X$ is the set of all possible histories $h$ of some fixed length $N$, i.e. $X = \lbrace h : h ∈ O^N \rbrace$, where $O$ is the set of outcomes. Note that since we use an independent infradistribution for each action, these histories only include observations, which is different from the usual definition of a history that includes also actions. For simplicity, the following description focuses on the bandit case, i.e. $O = \lbrace \text{no reward}, \text{reward} \rbrace$. The framework and code work for general finite outcome sets.

We call a **pure measure** one that corresponds to a definite reward probability $p$. In this case, we can easily construct the probability measure. For a history in which we got a reward $A$ times and did not get a reward $B$ times, the probability of that history is $p^A (1-p)^B$. Applying this to all histories of length $N$, we obtain the full measure.

A **mixed measure** is a linear combination of pure measures, i.e. $Σ_i c_i μ_i$. Note that a mixed measure no longer corresponds to a definite reward probability $p$. By mixing, we lose information about which probabilities and mixing coefficients were used.

As described, the computational complexity of this procedure scales exponentially with the history length. The actual code parametrises measures in a such way that components can be constructed on-the-fly in constant time. Effectively, we only write out those histories that actually matter. The idea is to store the probabilities of the pure measures and the mixing coefficients to construct elements of the measure at greater depths as needed. See `ibrl/infrabayesian/a_measure.py` for implementation details. Conceptually, this approach is equivalent to mapping out all the histories for some very large $N$, as described above. 

The actual `AMeasure` object in the code maintains only the fixed values $(λμ,b)$. This object does not get updated during learning. It only provides a method to compute the probability distribution $p(o|h)$ over outcomes $o$ given some observed history $h$. In particular

$$p(o|h) = \frac{μ(h + o)}{μ(h)}$$

Where $h+o$ means appending observation $o$ to history $h$. In the code, histories are encoded as integer arrays that indicate how often each outcome has occurred (the numbers $A$ and $B$ in the above example).

The expectation value of the a-measure under a given reward function $f$ and for a given history $h$ is then computed as:

$$a(f|h) = λ \sum_o p(o|h) f(o) + b$$

In the code, we can construct pure a-measures using `AMeasure.pure(p,λ,b)` where `p`= $p(o)$ is a probability distribution over outcomes. Mixed a-measures are constructed as `AMeasure.mixed(ps,cs,λ,b)`, where `ps`= $p_i(o)$ is a set of probability distributions and `cs`= $c_i$ are the associated mixing coefficients.

### Infradistributions
Infradistributions are represented by their extremal minimal points. The `Infradistribution` object also stores the `history` observed so far (as an integer array, see above).

The infra-expected value $E_H[f]$ of an infradistribution $H$ and reward function $f$ is computed as

$$E_H[f] = \min_{a ∈ H} a(f|h)$$

using the history $h$ stored in the object.

The [IB update rule](https://www.lesswrong.com/posts/YAa4qcMyoucRS2Ykr/basic-inframeasure-theory#Definition_11__Updating) reads for each a-measure

$$(λμ,b) → (λμ L, b + λμ(0 ★^L g)) → \frac{(λμ L, b + λμ(0 ★^L g) - E_H[0 ★^L g])}{E_H[1 ★^L g] - E_H[0 ★^L g]}$$

where $L$ indicates the observed event, $g$ is the reward function and $f ★^L g = Lf + (1-L)g$ is the gluing operator. The first arrow is the raw update and the second arrow is the renormalisation step. The raw update truncates the probability measure to the observed branch of history ($λμL$) adds the off-history reward to the offset term ($b+λμ$). The renormalisation step ensures that $E_H[0] = 0$ and $E_H[1] = 1$.

Most of these steps work as written. Only the truncation step requires some attention due to the way that measures are encoded. As described above, a-measures merely evaluate probabilities/expectation values given a certain history. Therefore, truncating the measure is equivalent to extending that history. Probabilities are computed for all possible events starting from a given history. For a longer history, some parts of these observations have already been excluded, so the measure is effectively truncated. In code, the truncation step looks like this:
```python
# Infradistribution.update
for a_measure in a_measures:
    a_measure.scale *= a_measure.probabilities(h)[o]
h[o] += 1
```
First, we multiply the scales λ by $p(o|h)$. Since the probability measure is always normalised, we need to keep track of the probability of the current history. This happens separately for every a-measure. Then we update the history object that is stored in the infradistribution. Note that this must happen after updating the scales, since otherwise we would be updating with $p(o|h+o)$.

We can compute **mixtures of infradistributions**, by mixing their a-measures. This step is somewhat more involved that the mixture of a-measures described above, because infradistributions might contain multiple a-measures and these a-measures might themselves already be mixtures. Infradistribution mixtures are computed by taking all combinations of exactly one a-measure from each input infradistributions. If these are mixed a-measures, they are split into pure a-measures. All of these pure a-measures are then combined into a single mixed a-measure, while taking into account both the mixing coefficients of the original mixed a-measures and the mixing coefficients of the infradistribution mixture. See `Infradistribution.mix` for details.

We can also mix infradistributions using KU. In this case we just merge the sets of a-measures.

In the code, infradistributions can be constructed from a list of a-measures as `Infradistribution(list_of_measures)`. A classical mixture of infradistributions is constructed as `Infradistribution.mix(list_of_infradistributions, mixing_coefficients)`. An infradistribution representing KU between several infradistributions can be created using `Infradistribution.mixKU(list_of_infradistributions)`.

### IB mode of operation
In the IB mode, we replace the second step of the initialisation above. We can use arbitrary combinations of classical priors and KU, including multiple levels of nesting. Let `m1`, `m2` and `m3` be three a-measures. Let `prior` be a probability distribution over two elements. Here are some examples for possible infradistribution initialisations:
```python
# No prior, no KU (i.e. no uncertainty)
Infradistribution([m1])

# Prior, no KU (classical agent)
Infradistribution.mix([
    Infradistribution([m1]),
    Infradistribution([m2])
], prior)

# No prior, KU
Infradistribution([m1,m2])

# Prior over KU
Infradistribution.mix([
    Infradistribution([m1]),
    Infradistribution([m2,m3])
], prior)

# KU over prior
Infradistribution.mixKU([
    Infradistribution.mix([
        Infradistribution([m1]),
        Infradistribution([m2])
    ], prior),
    Infradistribution([m3])
])
```

We see that we can combine classical priors and KU in arbitrary ways. Some combinations of infradistributions can be expressed in different ways. For example, we have `prior(m1, KU(m2,m3)) = KU(prior(m1,m2), prior(m1,m3))`, where `prior(a,b)` and `KU(a,b)` are the infradistributions that result from mixing `a` and `b` according to some classical prior or KU respectively. More examples are given in `experiments/ib_architecture/ib_mixtures.ipynb`.


## TODO / Questions
- Actually test IB agent!!
- Think about policy dependence
- Can we handle correlations between actions? Is having one infradistribution per action the correct approach? Histories usually include actions + observations. Here we only use observations.
