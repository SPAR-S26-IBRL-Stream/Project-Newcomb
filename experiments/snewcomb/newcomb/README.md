# Newcomb InfraBayesian Experiment

## What this experiment does

Test of InfraBayesian agent in the Newcomb environment to help understand the models better. 

It does three simple things:

- builds the standard Newcomb setup from `ibrl.environments`
- gives the agent a `NewcombLikeBelief`
- runs the interaction loop with `simulate()`

## Setup / How to run

From the project root:

```bash
uv run python experiments/syed/newcomb/newcomb_ib.py
```

## Design decisions

- Reuse `InfraBayesianAgent` because the point here is to test the existing agent.
- Use `NewcombLikeBelief` because this environment depends on both the agent's action and the predictor's action.
- Use `simulate()` because that is already the standard interaction loop in the repo, and well, because it keeps the experiment short.

## Results

Observed output from a local run:

```text
Newcomb InfraBayesian experiment
optimal_reward=10.000
mean_reward=7.720
final_step_reward=8.000
final_policy=[0.59 0.41]
```

The main number I care about here is `final_policy ≈ [0.59, 0.41]`.

That means the learned policy puts a bit more weight on action `0` than action `1` by the end of the run. In this setup, that looks like the agent leaning toward one-boxing, but just a bit. 

## Next steps / ideas

- vary predictor accuracy if we add that to the environment setup
- vary the box rewards and see when the policy shifts
- compare against other agents already in `ibrl.agents`
- try longer runs to see if the policy becomes more decisive
