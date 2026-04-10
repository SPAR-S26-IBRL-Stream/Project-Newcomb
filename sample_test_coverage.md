# Before adding new test files test_infrabayesian_beliefs.py and test_knightian_uncertainty.py



| Name | Stmts | Miss | Cover | Missing |
|---|---|---|---|---|
| ibrl/__init__.py | 8 | 0 | 100% | |
| ibrl/agents/__init__.py | 9 | 0 | 100% | |
| ibrl/agents/base.py | 21 | 2 | 90% | 35, 63 |
| ibrl/agents/base_greedy.py | 42 | 14 | 67% | 26, 44-46, 60-64, 73-79 |
| ibrl/agents/bayesian.py | 17 | 1 | 94% | 32 |
| ibrl/agents/exp3.py | 26 | 1 | 96% | 47 |
| ibrl/agents/experimental1.py | 11 | 5 | 55% | 15-19 |
| ibrl/agents/experimental2.py | 41 | 31 | 24% | 28-33, 36-54, 57-70, 73-76, 79 |
| ibrl/agents/experimental3.py | 22 | 13 | 41% | 24-28, 32-37, 40, 43 |
| ibrl/agents/q_learning.py | 23 | 3 | 87% | 29-30, 42 |
| ibrl/environments/__init__.py | 10 | 0 | 100% | |
| ibrl/environments/asymmetric_damascus.py | 6 | 1 | 83% | 13 |
| ibrl/environments/bandit.py | 11 | 0 | 100% | |
| ibrl/environments/base.py | 22 | 2 | 91% | 61, 71 |
| ibrl/environments/base_newcomb_like.py | 18 | 0 | 100% | |
| ibrl/environments/coordination.py | 6 | 1 | 83% | 12 |
| ibrl/environments/damascus.py | 6 | 0 | 100% | |
| ibrl/environments/newcomb.py | 6 | 0 | 100% | |
| ibrl/environments/policy_dependent_bandit.py | 9 | 3 | 67% | 13, 16-17 |
| ibrl/environments/switching.py | 24 | 13 | 46% | 17-19, 23-30, 33, 36-40 |
| ibrl/simulators/__init__.py | 2 | 0 | 100% | |
| ibrl/simulators/simulator.py | 37 | 2 | 95% | 45, 63 |
| ibrl/utils/__init__.py | 3 | 0 | 100% | |
| ibrl/utils/construction.py | 35 | 0 | 100% | |
| ibrl/utils/debug.py | 4 | 1 | 75% | 9 |
| ibrl/utils/sampling.py | 6 | 0 | 100% | |
| TOTAL | 425 | 93 | 78% | |

------------------------------
Summary Results: 39 passed in 0.36s


# After adding new test files test_infrabayesian_beliefs.py and test_knightian_uncertainty.py

test_infra_basic test coverage


| Module | Statements | Missing | Coverage | Uncovered Lines |
|--------|-----------|---------|----------|-----------------|
| ibrl/__init__.py | 8 | 0 | 100% | — |
| ibrl/agents/__init__.py | 11 | 0 | 100% | — |
| ibrl/agents/base.py | 22 | 2 | 91% | 37, 65 |
| ibrl/agents/base_greedy.py | 42 | 11 | 74% | 26, 44-46, 60-64, 76-79 |
| ibrl/agents/bayesian.py | 18 | 1 | 94% | 33 |
| ibrl/agents/bernoulli_bayesian.py | 20 | 1 | 95% | 32 |
| ibrl/agents/exp3.py | 27 | 1 | 96% | 48 |
| ibrl/agents/experimental1.py | 11 | 5 | 55% | 15-19 |
| ibrl/agents/experimental2.py | 42 | 32 | 24% | 28-33, 36-54, 57-71, 74-77, 80 |
| ibrl/agents/experimental3.py | 22 | 13 | 41% | 24-28, 32-37, 40, 44 |
| ibrl/agents/infrabayesian.py | 36 | 5 | 86% | 61, 75-78 |
| ibrl/agents/q_learning.py | 24 | 3 | 88% | 30-31, 43 |
| ibrl/environments/__init__.py | 11 | 0 | 100% | — |
| ibrl/environments/asymmetric_damascus.py | 6 | 1 | 83% | 13 |
| ibrl/environments/bandit.py | 11 | 0 | 100% | — |
| ibrl/environments/base.py | 27 | 2 | 93% | 67, 77 |
| ibrl/environments/base_newcomb_like.py | 17 | 0 | 100% | — |
| ibrl/environments/bernoulli_bandit.py | 15 | 1 | 93% | 28 |
| ibrl/environments/coin_tossing_game_envs.py | 9 | 9 | 0% | 1-26 |
| ibrl/environments/coordination.py | 6 | 1 | 83% | 12 |
| ibrl/environments/damascus.py | 6 | 0 | 100% | — |
| ibrl/environments/newcomb.py | 6 | 0 | 100% | — |
| ibrl/environments/policy_dependent_bandit.py | 9 | 3 | 67% | 13, 16-17 |
| ibrl/environments/switching.py | 24 | 13 | 46% | 17-19, 23-30, 33, 36-40 |
| ibrl/infrabayesian/__init__.py | 4 | 0 | 100% | — |
| ibrl/infrabayesian/a_measure.py | 15 | 2 | 87% | 21, 28 |
| ibrl/infrabayesian/beliefs.py | 85 | 19 | 78% | 22, 32, 43, 48, 66, 88, 101-103, 106-110, 113, 124, 130-134 |
| ibrl/infrabayesian/infradistribution.py | 41 | 3 | 93% | 52, 61, 93 |
| ibrl/outcome.py | 5 | 0 | 100% | — |
| ibrl/simulators/__init__.py | 2 | 0 | 100% | — |
| ibrl/simulators/simulator.py | 37 | 2 | 95% | 45, 63 |
| ibrl/utils/__init__.py | 3 | 0 | 100% | — |
| ibrl/utils/construction.py | 49 | 10 | 80% | 35-36, 87-95 |
| ibrl/utils/debug.py | 4 | 1 | 75% | 9 |
| ibrl/utils/sampling.py | 6 | 0 | 100% | — |
| **TOTAL** | **681** | **141** | **79%** | — |

**Test Results:** 82 passed, 2 warnings


## After the making the changes acc to the PR review(both human devs and claude's) by in https://github.com/SPAR-S26-IBRL-Stream/Project-Newcomb/pull/25


| Name | Stmts | Miss | Cover | Missing |
|------|-------|------|-------|---------|
| ibrl/__init__.py | 8 | 0 | 100% | |
| ibrl/agents/__init__.py | 11 | 0 | 100% | |
| ibrl/agents/base.py | 22 | 2 | 91% | 37, 65 |
| ibrl/agents/base_greedy.py | 42 | 11 | 74% | 26, 44-46, 60-64, 76-79 |
| ibrl/agents/bayesian.py | 18 | 1 | 94% | 33 |
| ibrl/agents/bernoulli_bayesian.py | 20 | 1 | 95% | 32 |
| ibrl/agents/exp3.py | 27 | 1 | 96% | 48 |
| ibrl/agents/experimental1.py | 11 | 5 | 55% | 15-19 |
| ibrl/agents/experimental2.py | 42 | 32 | 24% | 28-33, 36-54, 57-71, 74-77, 80 |
| ibrl/agents/experimental3.py | 22 | 13 | 41% | 24-28, 32-37, 40, 44 |
| ibrl/agents/infrabayesian.py | 36 | 5 | 86% | 61, 75-78 |
| ibrl/agents/q_learning.py | 24 | 1 | 96% | 43 |
| ibrl/environments/__init__.py | 11 | 0 | 100% | |
| ibrl/environments/asymmetric_damascus.py | 6 | 1 | 83% | 13 |
| ibrl/environments/bandit.py | 11 | 0 | 100% | |
| ibrl/environments/base.py | 27 | 2 | 93% | 67, 77 |
| ibrl/environments/base_newcomb_like.py | 17 | 0 | 100% | |
| ibrl/environments/bernoulli_bandit.py | 15 | 1 | 93% | 28 |
| ibrl/environments/coin_tossing_game_envs.py | 9 | 9 | 0% | 1-26 |
| ibrl/environments/coordination.py | 6 | 1 | 83% | 12 |
| ibrl/environments/damascus.py | 6 | 0 | 100% | |
| ibrl/environments/newcomb.py | 6 | 0 | 100% | |
| ibrl/environments/policy_dependent_bandit.py | 9 | 3 | 67% | 13, 16-17 |
| ibrl/environments/switching.py | 24 | 13 | 46% | 17-19, 23-30, 33, 36-40 |
| ibrl/infrabayesian/__init__.py | 4 | 0 | 100% | |
| ibrl/infrabayesian/a_measure.py | 15 | 2 | 87% | 21, 28 |
| ibrl/infrabayesian/beliefs.py | 85 | 19 | 78% | 22, 32, 43, 48, 66, 88, 101-103, 106-110, 113, 124, 130-134 |
| ibrl/infrabayesian/infradistribution.py | 41 | 3 | 93% | 52, 61, 93 |
| ibrl/outcome.py | 5 | 0 | 100% | |
| ibrl/simulators/__init__.py | 2 | 0 | 100% | |
| ibrl/simulators/simulator.py | 37 | 2 | 95% | 45, 63 |
| ibrl/utils/__init__.py | 3 | 0 | 100% | |
| ibrl/utils/construction.py | 49 | 10 | 80% | 35-36, 87-95 |
| ibrl/utils/debug.py | 4 | 1 | 75% | 9 |
| ibrl/utils/sampling.py | 6 | 0 | 100% | |
| **TOTAL** | **681** | **139** | **80%** | |

**Test Results:** 78 passed, 2 warnings in 1.01s





