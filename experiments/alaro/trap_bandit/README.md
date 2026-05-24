# Trap Bandit Reproduction

This experiment reproduces the trap-bandit robustness figures and summary table
used in **Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL
For Worst-Case Robustness**. The checked-in output directory is:

```text
experiments/alaro/trap_bandit/results_report_200_pcat001/
```

## Reproduce Outputs for Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL For Worst-Case Robustness

Run from the repository root:

```bash
uv run python -m experiments.alaro.trap_bandit.run
```

The defaults are the settings used for
**Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL For
Worst-Case Robustness**:

```text
num_worlds = 200
num_steps = 100
p_cat = 0.01
p_low = 0.3
p_high = 0.7
bootstrap_samples = 5000
output_dir = experiments/alaro/trap_bandit/results_report_200_pcat001
```

The script writes:

| output | use in Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL For Worst-Case Robustness |
| --- | --- |
| `mostly_risky_prior_comparison_grid.png` | Mostly-risky DGP comparison figure |
| `mostly_safe_prior_comparison_grid.png` | Mostly-safe DGP comparison figure |
| `balanced_prior_comparison_grid.png` | Balanced DGP comparison figure |
| `results_table.md` | Final cumulative expected-regret percentile table |
| `*_summary.json` | Per-condition regret, trapped-arm pull rate, and catastrophe summaries |
| `*_bootstrap_summary.json` | Bootstrap confidence intervals for the table |
| `config.json` | Full run configuration used for the output directory |

`results.md` contains the experiment description, embedded checked-in figures,
and the summary table text.

## Smoke Run

For a quick end-to-end check without regenerating the full reproduction outputs:

```bash
uv run python -m experiments.alaro.trap_bandit.run \
  --num-worlds 2 \
  --num-steps 10 \
  --bootstrap-samples 0 \
  --output-dir /tmp/project-newcomb-trap-bandit-smoke
```

Smoke-run results are not comparable to the figures in
**Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL For
Worst-Case Robustness**.

## Agents Compared

The script compares:

- `bayes_greedy`: Bayesian agent with greedy action selection.
- `bayes_thompson`: Bayesian agent with posterior-component Thompson sampling.
- `bayes_ucb`: Bayesian agent with posterior action-value quantile UCB.
- `ib`: infra-Bayesian agent with Knightian safe-vs-risky uncertainty.

The Bayesian agents use point priors over safe-vs-risky world type. The
infra-Bayesian agent does not use a safe-vs-risky point prior; it keeps that
uncertainty Knightian.
