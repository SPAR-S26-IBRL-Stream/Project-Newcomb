# Experiments

This directory contains exploratory work and experiments for
**Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL For
Worst-Case Robustness**. For review or reproduction, start with the entries
below rather than the individual notebooks, which may contain exploratory work.

## Reproduction Index

Run commands from the repository root after installing the project dependencies.

| item in **Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL For Worst-Case Robustness** | experiment | command | outputs |
| --- | --- | --- | --- |
| Classical-equivalence validation figure | `experiments/ib_architecture` | `uv run python -m experiments.ib_architecture.validate_classical` | `experiments/ib_architecture/validate_classical.png` |
| Knightian-uncertainty worst-case robustness figure | `experiments/ib_architecture` | `uv run python -m experiments.ib_architecture.ku_demo` | `experiments/ib_architecture/ku_demo_outputs/` |
| Newcomb predictor-accuracy figure | `experiments/ib_architecture` | `uv run python -m experiments.ib_architecture.newcomb_accuracy` | `experiments/ib_architecture/newcomb_accuracy_outputs/` |
| Trap-bandit robustness figures and summary table | `experiments/alaro/trap_bandit` | `uv run python -m experiments.alaro.trap_bandit.run` | `experiments/alaro/trap_bandit/results_report_200_pcat001/` |

The output files and directories listed above are checked in with the figures
and summaries used by **Infra-Bayesian Reinforcement Learning Agents Outperform
Classical RL For Worst-Case Robustness**. Re-running the commands regenerates
those artifacts with the settings used for that work.
