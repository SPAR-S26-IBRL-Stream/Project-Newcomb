# Experiments

This directory contains exploratory work and experiments for
**Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL For
Worst-Case Robustness**. For review or reproduction, start with the entries
below rather than the individual notebooks, which may contain exploratory work.

## Reproduction Index

Run commands from the repository root after installing the project dependencies.

| item in **Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL For Worst-Case Robustness** | experiment | command | outputs |
| --- | --- | --- | --- |
| Classical-equivalence validation figure | `experiments/fllor2` | `uv run python -m experiments.fllor2.validate_classical` | `experiments/fllor2/validate_classical.png` |
| Trap-bandit robustness figures and summary table | `experiments/alaro/trap_bandit` | `uv run python -m experiments.alaro.trap_bandit.run` | `experiments/alaro/trap_bandit/results_report_200_pcat001/` |

The output files and directories listed above are checked in with the figures
and summaries used by **Infra-Bayesian Reinforcement Learning Agents Outperform
Classical RL For Worst-Case Robustness**. Re-running the commands regenerates
those artifacts with the settings used for that work.
