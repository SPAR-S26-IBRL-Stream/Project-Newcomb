# Infra-Bayesian Primer

This primer is for contributors working on this codebase's infrabayesian agent experiments. It summarizes the parts of infra-Bayesianism most relevant to implementation: uncertainty models, updating, decision rules, learning notions, and supra-POMDPs. It deliberately skips infra-Bayesian physicalism.

## Why infra-Bayesianism?

Classical Bayesian reinforcement learning represents uncertainty as a prior over precise environments. That works best in a **realizable** setting: the true environment is assumed to be in the model class. Infra-Bayesianism is designed for the **non-realizable** setting, where the agent may only have partial properties or coarse descriptions of reality rather than a full correct model. The original sequence frames this as a generalization of epistemology, decision theory, and reinforcement learning based on imprecise probability, aimed at prior misspecification, grain-of-truth, and non-realizability problems ([Diffractor and Kosoy, 2020](https://www.alignmentforum.org/posts/zB4f7QqKhBHa5b37a/introduction-to-the-infra-bayesianism-sequence); [LessWrong wiki](https://www.lesswrong.com/w/infra-bayesianism)).

The practical slogan is:

> A Bayesian hypothesis says "the world is this distribution." An infra-Bayesian hypothesis can say "the world satisfies this property or lies somewhere in this closed convex set of distributions."

This makes infra-Bayesian agents naturally conservative: they optimize against the worst compatible model, while still allowing learning and performance guarantees when the true world satisfies learnable properties.

## Core objects

### Credal sets

A **credal set** is a closed, convex set of probability distributions. It represents Knightian or imprecise uncertainty: the agent does not choose one exact distribution, but keeps a set of distributions compatible with its information. Credal sets are the simplest "crisp" special case of infradistributions ([Gelb, "Credal Sets and Infra-Bayes Learnability"](https://www.lesswrong.com/posts/rkhaRnAc6dLzQT2sJ/an-introduction-to-credal-sets-and-infra-bayes-learnability-1); [nLab](https://ncatlab.org/nlab/show/infra-Bayesianism)).

For a reward or utility function `U`, a crisp infra-Bayesian value is usually the worst-case expectation:

```text
V(C, U) = inf_{P in C} E_P[U]
```

For losses, signs are usually flipped:

```text
L(C, loss) = sup_{P in C} E_P[loss]
```

So for reward-maximization, the agent uses **maximin**:

```text
choose pi maximizing inf_{P compatible with pi} E_P[U]
```

For loss-minimization, it uses the analogous **minimax** rule.

### Infradistributions

A general **infradistribution** is more than a credal set. The original formalism represents it as a set of signed affine measures, or equivalently as a concave, monotone expectation-like functional on utility functions ([Diffractor, "Basic Inframeasure Theory"](https://www.lesswrong.com/posts/YAa4qcMyoucRS2Ykr/basic-inframeasure-theory); [AXRP interview with Kosoy](https://axrp.net/episode/2021/03/10/episode-5-infra-bayesianism-vanessa-kosoy.html)).

The extra machinery exists mainly because plain credal sets do not update dynamically consistently. A plain credal-set update that just Bayesian-updates each distribution can forget how much probability each distribution assigned to the observed branch and can forget off-branch utility. General infradistributions keep enough bookkeeping to make updates agree with the policy that was optimal before the observation.

For this project, a useful implementation hierarchy is:

1. Start with finite credal sets or polytopes when possible.
2. Use worst-case expectation for decisions.
3. Only introduce full affine/signed-affine machinery when dynamic consistency across updates matters for the experiment.

### Infrakernels

A probability kernel maps a state to a probability distribution over next states:

```text
K: X -> Delta(Y)
```

A crisp **infrakernel** maps a state to a credal set over next states:

```text
K: X -> Box(Y)
```

Here `Box(Y)` denotes the space of credal sets over `Y`. Every ordinary probability kernel is a special case: a singleton credal set. Infrakernels are the transition objects used in supra-MDPs and supra-POMDPs ([Gelb, "Credal Sets and Infra-Bayes Learnability"](https://www.lesswrong.com/posts/rkhaRnAc6dLzQT2sJ/an-introduction-to-credal-sets-and-infra-bayes-learnability-1); [Gelb, "Crisp Supra-Decision Processes"](https://www.lesswrong.com/posts/mt82ZhdEsfh6CNYse/crisp-supra-decision-processes)).

## Updating and dynamic consistency

Classical Bayes:

```text
prior P -> observe event E -> posterior P(. | E)
```

Naive credal-set update:

```text
C -> { P(. | E) : P in C and P(E) > 0 }
```

This is often not enough. If one distribution in `C` assigned the observation much higher probability than another, naive setwise conditioning can discard that relative weight. Worse, in sequential decision problems it can lead to **dynamic inconsistency**: after observing a branch, the updated agent may prefer a continuation policy that the original agent would reject ([Diffractor and Kosoy, "Introduction"](https://www.alignmentforum.org/posts/zB4f7QqKhBHa5b37a/introduction-to-the-infra-bayesianism-sequence); [AXRP interview](https://axrp.net/episode/2021/03/10/episode-5-infra-bayesianism-vanessa-kosoy.html)).

Infra-Bayesian updating addresses this by updating richer objects. In the affine-measure view, an element looks like:

```text
(m, b)
```

where `m` is a finite measure-like object and `b` is a constant term. Intuitively:

- `m` carries branch probability and expected utility on the observed branch.
- `b` carries value already secured or lost off the observed branch.

The update has two conceptual steps:

1. **Raw update:** restrict/chop each measure to the observed branch, preserving branch weight.
2. **Renormalize:** scale/shift the resulting set back into the normalized infradistribution form.

The point is not just better posterior beliefs; the point is that future-you's local decision after updating agrees with past-you's globally optimal plan. This is formalized as dynamic consistency in the belief-functions/decision-theory post ([Diffractor, "Belief Functions And Decision Theory"](https://www.greaterwrong.com/posts/e8qFDMzs2u9xf5ie6/belief-functions-and-decision-theory)).

## Decision theory

### Policies, histories, and causal laws

An environment plus a policy induces a distribution over complete interaction histories. Infra-Bayesianism packages sets of possible environments into **crisp causal laws**: maps from policies to credal sets over histories. In newer terminology, these take the role of hypotheses in infra-Bayesian reinforcement learning ([Gelb, "Credal Sets and Infra-Bayes Learnability"](https://www.lesswrong.com/posts/rkhaRnAc6dLzQT2sJ/an-introduction-to-credal-sets-and-infra-bayes-learnability-1)).

```text
Lambda: Policy -> CredalSet[History]
```

Given policy `pi` and loss `L`, the robust value is:

```text
E_{Lambda(pi)}[L] = sup_{P in Lambda(pi)} E_P[L]
```

The infra-Bayes optimal policy minimizes expected robust loss under a prior over causal laws. In reward form, it maximizes worst-case expected utility.

### Murphy

Many infra-Bayesian explanations use "Murphy" as shorthand for the worst-case selector inside a credal set. Murphy is not literally an adversary in the environment; it is the mathematical operation that chooses the compatible model with worst value for the current policy. Guarantees are therefore lower bounds: if the real environment is less hostile than the worst compatible one, the agent may do better.

### Policy-dependent problems

Infra-Bayesianism is notable because it can represent policy-dependent decision problems such as Newcomb-like setups, counterfactual mugging, and related UDT-style cases. The original sequence uses the "Nirvana trick" to encode policy-dependent environments as sets of ordinary environments: if the agent deviates from the hard-coded policy, it reaches a special high-value outcome, forcing the worst-case selector to pick an environment whose hard-coded policy matches the agent's actual policy ([Diffractor and Kosoy, "Introduction"](https://www.alignmentforum.org/posts/zB4f7QqKhBHa5b37a/introduction-to-the-infra-bayesianism-sequence); [Diffractor, "Belief Functions And Decision Theory"](https://www.greaterwrong.com/posts/e8qFDMzs2u9xf5ie6/belief-functions-and-decision-theory)).

Implementation note: experiments in this repo do not need to reproduce the full Nirvana formalism unless testing acausal/policy-selection behavior directly. For many supra-POMDP experiments, finite credal transition sets are enough.

## Learning and regret

Classical Bayesian RL asks whether a Bayes-optimal policy learns the true environment or achieves low regret against the best policy for that environment. Infra-Bayesianism instead asks whether the agent learns useful properties/hypotheses that reality satisfies, even if reality itself is not in the model class.

For a hypothesis `Theta`, utility family `U_gamma`, and policy `pi`, the infra-regret analogue is:

```text
R(pi, Theta, U) = max_{pi*} ( E_Theta(pi*)(U) - E_Theta(pi)(U) )
```

A family of hypotheses is **learnable** if there is a policy family whose regret goes to zero as the time discount becomes patient (`gamma -> 1`). An infra-Bayes-optimal policy for a suitable prior learns any learnable collection of hypotheses, paralleling the classical Bayesian result ([Diffractor, "Belief Functions And Decision Theory"](https://www.greaterwrong.com/posts/e8qFDMzs2u9xf5ie6/belief-functions-and-decision-theory); [Gelb, "Credal Sets and Infra-Bayes Learnability"](https://www.lesswrong.com/posts/rkhaRnAc6dLzQT2sJ/an-introduction-to-credal-sets-and-infra-bayes-learnability-1)).

For experiments, this suggests measuring:

- worst-case value under the credal model,
- realized value in sampled concrete environments,
- regret against the best policy that knew the true member of the environment class,
- robustness when the Bayesian agent's precise prior is misspecified.

## Supra-MDPs and supra-POMDPs

The supra-POMDP approach is the most directly experiment-ready part of the theory.

An ordinary MDP has precise transition probabilities:

```text
T: S x A -> Delta(S)
```

A **crisp supra-MDP** replaces the transition kernel with an infrakernel:

```text
T: S x A -> Box(S)
```

So each state-action pair maps to a credal set of possible next-state distributions. A **crisp supra-POMDP** adds partial observability:

```text
(S, Theta_0, A, O, T, B, L, gamma)
```

where `Theta_0` is an initial credal set over states, `T` is a transition infrakernel, `B` maps states to observations, `L` is loss, and `gamma` is discount ([Gelb, "Crisp Supra-Decision Processes"](https://www.lesswrong.com/posts/mt82ZhdEsfh6CNYse/crisp-supra-decision-processes)).

Key facts for this project:

- A supra-MDP differs from an MDP only by making transitions set-valued.
- A supra-POMDP can represent a crisp causal law, and crisp causal laws can be translated back into supra-POMDPs.
- Coarsening a precise MDP's state space can naturally produce a supra-MDP: collapsed states inherit a convex set of transition distributions from the detailed states they hide.
- Finite/polytopic supra-MDPs can often be treated like robust MDPs or single-controller zero-sum stochastic games, where the agent chooses actions and Murphy chooses transition distributions.

This gives a clean experimental template:

1. Build a ground-truth MDP or POMDP.
2. Coarsen or hide part of its state.
3. Give the infra agent the resulting credal transition model.
4. Compare it to Bayesian agents that must choose a precise prior or point estimate over hidden dynamics.

Infra-Bayesianism should shine when the coarse property is enough to act safely or well, but a precise Bayesian model is brittle under misspecification.

## Minimal implementation checklist

For a finite experiment, represent each credal transition set as either:

- interval probabilities for small binary cases,
- a list of vertices of a convex polytope,
- or a sampler/adversarial optimizer over allowed transition distributions.

Then implement:

1. **Value backup**

   ```text
   V_t(s) = min_a [ L(s,a) + gamma * sup_{p in T(s,a)} E_{s'~p} V_{t+1}(s') ]
   ```

   Use `max` instead of `min` and `inf` instead of `sup` if coding rewards rather than losses.

2. **Policy extraction**

   Pick the action that optimizes the robust Bellman backup.

3. **Belief/state update**

   For crisp finite supra-POMDPs, maintain a credal set over hidden states and update by propagating each compatible distribution through the selected action and observed outcome, then taking closure/convex hull. Treat this as an approximation to full infra-updating unless affine off-branch terms are explicitly modeled.

4. **Comparison baseline**

   Use one or more classical Bayesian agents with precise priors. Stress-test by changing the ground-truth member of the credal set or by choosing a coarse-state refinement not represented by the Bayesian prior.

## References

- Diffractor and Vanessa Kosoy, [Introduction To The Infra-Bayesianism Sequence](https://www.alignmentforum.org/posts/zB4f7QqKhBHa5b37a/introduction-to-the-infra-bayesianism-sequence)
- Diffractor, [Basic Inframeasure Theory](https://www.lesswrong.com/posts/YAa4qcMyoucRS2Ykr/basic-inframeasure-theory)
- Diffractor, [Belief Functions And Decision Theory](https://www.greaterwrong.com/posts/e8qFDMzs2u9xf5ie6/belief-functions-and-decision-theory)
- Brittany Gelb, [An Introduction to Credal Sets and Infra-Bayes Learnability](https://www.lesswrong.com/posts/rkhaRnAc6dLzQT2sJ/an-introduction-to-credal-sets-and-infra-bayes-learnability-1)
- Brittany Gelb, [Crisp Supra-Decision Processes](https://www.lesswrong.com/posts/mt82ZhdEsfh6CNYse/crisp-supra-decision-processes)
- Daniel Filan and Vanessa Kosoy, [AXRP Episode 5: Infra-Bayesianism with Vanessa Kosoy](https://axrp.net/episode/2021/03/10/episode-5-infra-bayesianism-vanessa-kosoy.html)
- [nLab: infra-Bayesianism](https://ncatlab.org/nlab/show/infra-Bayesianism)
