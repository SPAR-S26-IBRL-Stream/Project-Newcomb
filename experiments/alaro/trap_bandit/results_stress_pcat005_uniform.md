# Trap Bandit Stress Experiment

Below we describe a stress variant of the trap-bandit experiment. Relative to the baseline run, this version raises catastrophe probability and uses a uniform arm-reward DGP to make high-reward/high-risk arms more tempting and mistakes more visible.

The details of our experiment are as follows. There are `K=2` possible arms to pull. There is a probability `alpha` of being in a risky world, and probability `1 - alpha` of being in a safe world.

At the beginning of a new run, p_1 and p_2 are newly sampled from a uniform distribution over valid non-catastrophe reward probabilities. The world_type = {risky, safe} is also sampled. In the safe world, each arm is Bernoulli and has fixed probability, `p_i`, of yielding reward `1`. In the risky world, the arm with the higher realized bias `p_i` is a three-sided die with a small probability `p_catastrophe` of yielding reward `-1000`; with probability `p_i`, it yields reward `1`; otherwise it yields reward `0`. The arm with the lower realized bias is still Bernoulli with reward = {1,0}.

```text
For each new run:
    sample alpha from the data-generating Beta distribution
    sample p1, p2 ~ Uniform(0, 1 - p_cat)
    sample world_type ~ Bernoulli(alpha)

    if safe world:
        arm i -> Bernoulli(p_i)

    if risky world:
        trapped_arm = argmax(p1, p2)
        trapped_arm -> reward -1000 (catastrophe) with probability 0.05
                        reward 1 with probability p_i
                        reward 0 otherwise
        other arm   -> Bernoulli(p_i)
```
Schema 1. Experiment world design.

We compare classical Bayesian agents and an infra-Bayesian agent using the same joint hypothesis machinery. Bayesian agents always use `Infradistribution.mix(...)`; the infra-Bayesian agent uses Knightian uncertainty over the safe-vs-risky world families via `Infradistribution.mixKU(...)`, while remaining classical/Bayesian (employing `Infradistribution.mix(...)`) over `p1,p2` within each family.

The Bayesian agent does not represent a full Beta prior over `alpha`. Instead, it receives a fixed point prior `P(risky) = E[alpha]` for the safe-vs-risky mixture: `0.5` for the correct condition, `2/7` for the mildly misspecified condition, `1/100` for the severely misspecified and mostly-safe conditions, and `99/100` for the severely pessimistic condition. This is because each agent acts within a single world, where `alpha` only induces the prior probability that the current world is risky; the variance of a population-level Beta prior over `alpha` would matter only for learning across many independently sampled worlds. By contrast, uncertainty over `p1,p2` is represented explicitly by a finite grid and updated from within-run observations. In this stress run, the classical `p1,p2` prior is changed to match the uniform DGP.

In the first experiment, the Bayesian point prior on `P(risky)` matches the data-generating process in expectation, and the agent's `p1,p2` prior matches the data-generating distribution. In the next experiments, we run misspecified point-prior conditions where Bayesian agents put too little or too much probability on the risky world. Finally, in the mostly-safe experiment, we change the data-generating process to be mostly safe, such that the expected value maximizer would risk pulling the higher-reward arm. The infra-Bayesian agent always shares the same classical `p1,p2` prior as the Bayesian agent but maintains Knightian uncertainty over whether the world is safe or risky.

For Bayesian agents, we compare three exploration strategies:

- greedy,
- Thompson sampling,
- empirical UCB.

For the infra-Bayesian agent, we use greedy action selection over its robust lower values, with uniform tie-breaking.

Regret is measured against the best policy with full knowledge of the true world. We report cumulative expected regret percentiles and trapped-arm pull-rate percentiles.

## Results

The implementation is in `experiments/alaro/trap_bandit/` and the results were generated using the below configs:

```text
num_worlds = 100
num_steps = 200
num_grid = 7
p_cat = 0.05
p_beta = (1, 1)
```

Each result figure has six subplots. Columns are `log(1 + cumulative expected regret)` and `argmax(p1,p2)` pull rate. Rows are overall average, safe worlds, and risky worlds.

![Correct-prior grid](results_stress_pcat005_uniform/correct_grid.png)

Figure 2a. Correct-prior results.

In the first experiment, the bayesian agent with a correctly specified prior has very similar behavior to the infra-bayesian agent, which maintains knightian uncertainty on whether it is in a risky world or not. They behave nearly identically in this setting because it is not favorable under this data generating process for an expected value maximizer to pull the risky arm. A key positive finding is that the infra-bayesian learner does properly learn which of the two arms is the risky one, at which point it can begin to behave safely. Notably, both non-greedy exploration strategies show significant regret in the risky worlds. 

Next, we examine two misspecified point priors for the probability that the world is risky.

![Misspecified-prior grid](results_stress_pcat005_uniform/misspecified_grid.png)

Figure 2b. Misspecified-prior results.

In the first, slightly misspecified setting, the Bayesian agent uses point prior `P(risky)=2/7`, while the data-generating process has `E[alpha]=1/2`. Results between the Bayesian and infra-Bayesian agents diverge slightly but not significantly.

![Severely misspecified-prior grid](results_stress_pcat005_uniform/severely_misspecified_grid.png)

Figure 2c. Severely misspecified-prior results.

However, in the extremely misspecified setting, the Bayesian agent uses point prior `P(risky)=1/100`, while the data-generating process has `E[alpha]=1/2`. The Bayesian agent incurs significant regret by pulling the risky arm until it adjusts its posterior enough to reflect the actual world and begins to act more conservatively.

The severely pessimistic comparison uses Bayesian point prior `P(risky)=99/100` with the same data-generating process `E[alpha]=1/2`. This condition tests whether IB's advantage in the severely misspecified setting is genuinely about not needing to choose a pessimism level in advance, rather than merely showing that any sufficiently pessimistic expected-value agent behaves safely.

![Severely pessimistic-prior grid](results_stress_pcat005_uniform/severely_pessimistic_grid.png)

Figure 2d. Severely pessimistic-prior results.

In this condition, greedy Bayes and IB behave nearly identically because the Bayesian point prior is pessimistic enough that the risky branch dominates early action values. This supports the cleaner interpretation of IB's advantage: a sufficiently pessimistic Bayesian prior can also be safe, but IB does not require selecting that pessimism level in advance.

Finally, we change the data-generating process to be mostly safe, with `E[alpha]=1/100`, and show the results below.

![Mostly-safe correct-prior grid](results_stress_pcat005_uniform/mostly_safe_correct_grid.png)

Figure 2e. Mostly-safe correctly specified prior results.

Here, the infra-bayesian agent can be seen to drastically underperform in cumulative regret because of course it is maintaining knightian uncertainty about the high reward arm being risky.

# Summary

In this stress setting, the under-pessimistic Bayesian agent's p95 regret is again worse than IB's in the severely misspecified condition, but the bootstrap intervals are still wide at `N=100`. The severely pessimistic condition again shows that a Bayesian point prior can match IB when pessimism is tuned high enough in advance. The mostly-safe condition remains the clearest robustness-cost comparison: greedy Bayes has lower p95 regret than IB, with tighter and non-overlapping intervals.

The stress DGP makes the environment harsher, but it does not fully remove tail-estimation noise at `N=100`: p95 intervals in the risky/misspecified comparisons remain wide. Treat this run as a stress check rather than a replacement for the baseline.

# Appendix

Final cumulative expected-regret percentiles from `results_stress_pcat005_uniform`. Brackets show 95% bootstrap CIs from 5000 resamples over worlds.

| condition | agent | catastrophe rate | p5, 95% CI | p50, 95% CI | p95, 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| correct | bayes_greedy | 0.11 | 0.00 [0.00, 0.00] | 49.19 [24.95, 49.51] | 663.39 [149.30, 2542.38] |
| correct | bayes_thompson | 0.53 | 3.39 [3.17, 4.29] | 49.94 [14.18, 223.49] | 1800.50 [1154.19, 2193.80] |
| correct | bayes_ucb | 0.53 | 5.36 [4.21, 7.72] | 99.34 [16.57, 398.98] | 1885.77 [1540.94, 2847.75] |
| correct | ib | 0.11 | 0.00 [0.00, 0.00] | 49.19 [24.95, 49.54] | 663.39 [149.12, 2542.38] |
| misspecified | bayes_greedy | 0.10 | 0.00 [0.00, 0.00] | 49.19 [24.95, 49.51] | 457.83 [149.30, 2542.38] |
| misspecified | bayes_thompson | 0.53 | 2.22 [1.40, 3.71] | 99.27 [7.86, 248.78] | 1985.67 [1356.03, 2299.89] |
| misspecified | bayes_ucb | 0.53 | 5.36 [4.21, 7.72] | 99.34 [16.57, 398.98] | 1885.77 [1540.94, 2847.75] |
| misspecified | ib | 0.11 | 0.00 [0.00, 0.00] | 49.19 [24.95, 49.54] | 663.39 [149.12, 2542.38] |
| severely misspecified | bayes_greedy | 0.23 | 0.00 [0.00, 0.00] | 0.00 [0.00, 31.04] | 1366.01 [751.84, 2542.38] |
| severely misspecified | bayes_thompson | 0.53 | 1.39 [0.49, 1.73] | 49.59 [8.42, 296.20] | 2098.92 [1409.11, 2760.75] |
| severely misspecified | bayes_ucb | 0.53 | 5.36 [4.21, 7.72] | 99.34 [16.57, 398.98] | 1885.77 [1540.94, 2847.75] |
| severely misspecified | ib | 0.11 | 0.00 [0.00, 0.00] | 49.19 [24.95, 49.54] | 663.39 [149.12, 2542.38] |
| severely pessimistic | bayes_greedy | 0.11 | 0.00 [0.00, 0.00] | 49.19 [24.95, 49.51] | 663.39 [149.30, 2542.38] |
| severely pessimistic | bayes_thompson | 0.21 | 1.46 [0.97, 16.91] | 143.53 [94.80, 197.68] | 503.05 [397.93, 841.80] |
| severely pessimistic | bayes_ucb | 0.53 | 5.36 [4.21, 7.72] | 99.34 [16.57, 398.98] | 1885.77 [1540.94, 2847.75] |
| severely pessimistic | ib | 0.11 | 0.00 [0.00, 0.00] | 49.19 [24.95, 49.54] | 663.39 [149.12, 2542.38] |
| mostly safe correct | bayes_greedy | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 3.24] | 124.70 [88.15, 135.26] |
| mostly safe correct | bayes_thompson | 0.00 | 1.13 [0.69, 1.22] | 3.04 [2.62, 3.55] | 10.61 [7.83, 11.80] |
| mostly safe correct | bayes_ucb | 0.00 | 2.33 [1.31, 4.60] | 12.36 [11.72, 13.31] | 17.61 [16.04, 18.68] |
| mostly safe correct | ib | 0.00 | 0.00 [0.00, 0.00] | 42.29 [18.91, 53.74] | 160.81 [127.98, 170.99] |
