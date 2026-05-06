# Trap Bandit Mostly-Risky Experiment

Below we describe a mostly-risky variant of the trap-bandit experiment. This run uses the original reward/catastrophe settings, but changes the first three conditions so that the data-generating process produces risky worlds with probability approximately `0.99`.

The details of our experiment are as follows. There are `K=2` possible arms to pull. There is a probability `alpha` of being in a risky world, and probability `1 - alpha` of being in a safe world.

At the beginning of a new run, p_1 and p_2 are newly sampled from a beta distribution. The world_type = {risky, safe} is also sampled.  In the safe world, each arm is Bernoulli and has fixed probability, `p_i`, of yielding reward `1`. In the risky world, the arm with the higher realized bias `p_i` is a three-sided die with a small probability `p_catastrophe` of yielding reward `-1000`; with probability `p_i`, it yields reward `1`; otherwise it yields reward `0`. The arm with the lower realized bias is still Bernoulli with reward = {1,0}.

```text
For each new run:
    sample alpha from the data-generating Beta distribution
    sample p1, p2 ~ Beta(2,2)
    sample world_type ~ Bernoulli(alpha)

    if safe world:
        arm i -> Bernoulli(p_i)

    if risky world:
        trapped_arm = argmax(p1, p2)
        trapped_arm -> reward -1000 (catastrophe) with probability 0.01
                        reward 1 with probability p_i
                        reward 0 otherwise
        other arm   -> Bernoulli(p_i)
```
Schema 1. Experiment world design.

We compare classical Bayesian agents and an infra-Bayesian agent using the same joint hypothesis machinery. Bayesian agents always use `Infradistribution.mix(...)`; the infra-Bayesian agent uses Knightian uncertainty over the safe-vs-risky world families via `Infradistribution.mixKU(...)`, while remaining classical/Bayesian (employing `Infradistribution.mix(...)`) over `p1,p2` within each family.

The Bayesian agent does not represent a full Beta prior over `alpha`. Instead, it receives a fixed point prior `P(risky) = E[alpha]` for the safe-vs-risky mixture: `0.99` for the correct mostly-risky condition, `0.5` for the moderately misspecified condition, `0.01` for the severely misspecified condition, and `0.01` for the mostly-safe condition. This is because each agent acts within a single world, where `alpha` only induces the prior probability that the current world is risky; the variance of a population-level Beta prior over `alpha` would matter only for learning across many independently sampled worlds. By contrast, uncertainty over `p1,p2` is represented explicitly by a finite grid and updated from within-run observations.

In the first experiment, the Bayesian point prior on `P(risky)` matches the mostly-risky data-generating process. In the next two experiments, Bayesian agents increasingly underestimate how likely risky worlds are. Finally, in the mostly-safe experiment, we keep the original mostly-safe comparison unchanged, such that the expected value maximizer would risk pulling the higher-reward arm. The infra-Bayesian agent always shares the same classical `p1,p2` prior as the Bayesian agent but maintains Knightian uncertainty over whether the world is safe or risky.

For Bayesian agents, we compare three exploration strategies:

- greedy,
- Thompson sampling,
- empirical UCB.

For the infra-Bayesian agent, we use greedy action selection over its robust lower values, with uniform tie-breaking.

Regret is measured against the best policy with full knowledge of the true world. We report cumulative expected regret percentiles and trapped-arm pull-rate percentiles.

## Results

The implementation is in `experiments/alaro/trap_bandit/` and the results were generated using the below configs:

```text
num_worlds = 200
num_steps = 100
num_grid = 7
p_cat = 0.01
p_beta = (2, 2)
condition_preset = mostly_risky
```

Each result figure has six subplots. Columns are `log(1 + cumulative expected regret)` and `argmax(p1,p2)` pull rate. Rows are overall average, safe worlds, and risky worlds.

![Correct-prior grid](results_mostly_risky_200_pcat001_grid7/correct_grid.png)

Figure 2a. Correct-prior results.

In the first experiment, the Bayesian agent has the correct point prior `P(risky)=0.99`. Greedy Bayes and IB behave identically in this run: both are already conservative enough to avoid much of the trapped-arm risk. Thompson sampling has lower p95 regret here, while UCB explores aggressively and has higher catastrophe rate.

Next, we examine two misspecified point priors for the probability that the world is risky.

![Misspecified-prior grid](results_mostly_risky_200_pcat001_grid7/misspecified_grid.png)

Figure 2b. Misspecified-prior results.

In the first misspecified setting, the Bayesian agent uses point prior `P(risky)=0.5`, while the data-generating process has `E[alpha]=0.99`. Greedy Bayes and IB still match closely, suggesting that this level of misspecification is not enough to change greedy action choice under these parameters.

![Severely misspecified-prior grid](results_mostly_risky_200_pcat001_grid7/severely_misspecified_grid.png)

Figure 2c. Severely misspecified-prior results.

However, in the extremely misspecified setting, the Bayesian agent uses point prior `P(risky)=0.01`, while the data-generating process has `E[alpha]=0.99`. Here greedy Bayes has a much higher catastrophe rate and median regret than IB, because it initially treats the trapped-arm hypothesis as very unlikely.

Finally, we change the data-generating process to be mostly safe, with `E[alpha]=1/100`, and show the results below.

![Mostly-safe correct-prior grid](results_mostly_risky_200_pcat001_grid7/mostly_safe_correct_grid.png)

Figure 2d. Mostly-safe correctly specified prior results.

Here, the infra-bayesian agent can be seen to drastically underperform in cumulative regret because of course it is maintaining knightian uncertainty about the high reward arm being risky.

# Summary

With the mostly-risky DGP, greedy Bayes and IB match when Bayes has the correct `P(risky)=0.99` prior or a moderately misspecified `P(risky)=0.5` prior. The strongest separation is in the severely misspecified condition: greedy Bayes has much higher catastrophe rate (`0.530` vs `0.190`) and median regret (`435.15` vs `19.05`) than IB. The p95 regret comparison does not favor IB in this run, because IB's conservative policy still pays high regret in some worlds relative to the full-knowledge optimal policy. The mostly-safe condition remains the clean robustness-cost case: greedy Bayes exploits more freely and has much lower regret than IB.

The useful story from this run is therefore not "IB wins every tail metric." It is more precise: IB is stable across safe-vs-risky prior misspecification, while greedy Bayes can suffer large median-regret and catastrophe-rate degradation when its point prior badly underestimates risky worlds.

# Appendix

Final cumulative expected-regret percentiles from `results_mostly_risky_200_pcat001_grid7`. Brackets show 95% bootstrap CIs from 5000 resamples over worlds.

| condition | agent | catastrophe rate | p5, 95% CI | p50, 95% CI | p95, 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| correct | bayes_greedy | 0.190 | 0.00 [0.00, 0.00] | 19.05 [9.86, 29.67] | 981.70 [949.74, 983.92] |
| correct | bayes_thompson | 0.185 | 18.56 [9.78, 19.62] | 79.50 [68.28, 97.21] | 497.89 [394.59, 579.11] |
| correct | bayes_ucb | 0.510 | 58.37 [38.66, 88.97] | 533.29 [456.72, 564.95] | 802.37 [762.22, 806.13] |
| correct | ib | 0.190 | 0.00 [0.00, 0.00] | 19.05 [9.86, 29.68] | 981.70 [953.03, 983.89] |
| misspecified | bayes_greedy | 0.195 | 0.00 [0.00, 0.00] | 19.05 [9.86, 29.80] | 981.70 [949.74, 983.92] |
| misspecified | bayes_thompson | 0.455 | 46.74 [19.93, 75.80] | 447.77 [418.26, 473.92] | 643.94 [598.81, 647.79] |
| misspecified | bayes_ucb | 0.510 | 58.37 [38.66, 88.97] | 533.29 [456.72, 564.95] | 802.37 [762.22, 806.13] |
| misspecified | ib | 0.190 | 0.00 [0.00, 0.00] | 19.05 [9.86, 29.68] | 981.70 [953.03, 983.89] |
| severely misspecified | bayes_greedy | 0.530 | 9.94 [0.00, 19.37] | 435.15 [325.34, 573.79] | 977.45 [962.26, 982.28] |
| severely misspecified | bayes_thompson | 0.555 | 49.92 [29.38, 99.95] | 558.28 [456.33, 603.16] | 904.12 [895.20, 916.63] |
| severely misspecified | bayes_ucb | 0.510 | 58.37 [38.66, 88.97] | 533.29 [456.72, 564.95] | 802.37 [762.22, 806.13] |
| severely misspecified | ib | 0.190 | 0.00 [0.00, 0.00] | 19.05 [9.86, 29.68] | 981.70 [953.03, 983.89] |
| mostly safe correct | bayes_greedy | 0.000 | 0.00 [0.00, 0.00] | 0.42 [0.34, 0.60] | 11.01 [8.55, 20.60] |
| mostly safe correct | bayes_thompson | 0.000 | 0.51 [0.27, 0.80] | 2.81 [2.28, 3.27] | 8.04 [6.46, 9.23] |
| mostly safe correct | bayes_ucb | 0.000 | 1.05 [0.59, 1.31] | 6.50 [5.45, 7.45] | 11.59 [10.95, 12.10] |
| mostly safe correct | ib | 0.000 | 0.00 [0.00, 0.00] | 16.46 [9.22, 20.87] | 66.38 [59.11, 70.15] |
