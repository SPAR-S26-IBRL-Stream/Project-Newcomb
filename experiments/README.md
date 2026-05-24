# Experiments

This directory contains exploratory work and experiments for
**Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL For
Worst-Case Robustness**. For review or reproduction, start with the entries
below rather than the individual notebooks, which may contain exploratory work.

## Reproduction Index

Run commands from the repository root after installing the project dependencies.

| paper item | experiment | command | outputs |
| --- | --- | --- | --- |
| Trap-bandit robustness figures and summary table | `experiments/alaro/trap_bandit` | `uv run python -m experiments.alaro.trap_bandit.run` | `experiments/alaro/trap_bandit/results_report_200_pcat001/` |

The trap-bandit output directory is checked in with the figures and summaries
used by the paper. Re-running the command above regenerates those artifacts
with the default paper settings.
