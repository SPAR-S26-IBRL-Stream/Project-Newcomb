# Experiments

This directory contains exploratory work and paper-facing experiments. For review
or reproduction, start with the entries below rather than the personal notebooks.

## Paper Figure Reproduction

Run commands from the repository root after installing the project dependencies.

| paper item | experiment | command | outputs |
| --- | --- | --- | --- |
| Trap-bandit robustness figures and summary table | `experiments/alaro/trap_bandit` | `uv run python -m experiments.alaro.trap_bandit.run` | `experiments/alaro/trap_bandit/results_report_200_pcat001/` |

The trap-bandit output directory is checked in with the figures and summaries
used by the current report. Re-running the command above regenerates those
artifacts with the default paper settings.

## Smoke Runs

Use smaller settings when checking that a script runs end-to-end:

```bash
uv run python -m experiments.alaro.trap_bandit.run \
  --num-worlds 2 \
  --num-steps 10 \
  --bootstrap-samples 0 \
  --output-dir /tmp/project-newcomb-trap-bandit-smoke
```

Smoke outputs are only for checking the command path; they are not the paper
figures.
