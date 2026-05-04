Name                                                         Stmts   Miss  Cover   Missing
------------------------------------------------------------------------------------------
ibrl/__init__.py                                                 8      0   100%
ibrl/agents/__init__.py                                         12      0   100%
ibrl/agents/base.py                                             21      2    90%   35, 63
ibrl/agents/base_greedy.py                                      41     14    66%   25, 43-45, 59-63, 72-78
ibrl/agents/bayesian.py                                         17      1    94%   32
ibrl/agents/bernoulli_bayesian.py                               19     11    42%   15, 18-22, 25-28, 31
ibrl/agents/discrete_bayesian.py                                23      1    96%   45
ibrl/agents/exp3.py                                             26      1    96%   47
ibrl/agents/experimental1.py                                    10      5    50%   14-18
ibrl/agents/experimental2.py                                    41     32    22%   27-32, 35-53, 56-70, 73-76, 79
ibrl/agents/experimental3.py                                    21     13    38%   23-27, 31-36, 39, 43
ibrl/agents/infrabayesian.py                                    81     23    72%   63, 74, 89, 97-105, 111-116, 146, 159-162, 166-174
ibrl/agents/policy_optimizer.py                                 44     18    59%   69-100
ibrl/agents/q_learning.py                                       23      1    96%   42
ibrl/agents/supra_pomdp_agent.py                                42     10    76%   33-34, 105-119
ibrl/environments/__init__.py                                   11      0   100%
ibrl/environments/asymmetric_damascus.py                         5      1    80%   12
ibrl/environments/bandit.py                                     10      0   100%
ibrl/environments/base.py                                       26      2    92%   66, 76
ibrl/environments/base_newcomb_like.py                          27      1    96%   24
ibrl/environments/bernoulli_bandit.py                           16      9    44%   16-17, 20-21, 24, 27-31
ibrl/environments/coin_tossing_game_envs.py                      9      9     0%   1-26
ibrl/environments/coordination.py                                5      1    80%   11
ibrl/environments/damascus.py                                    5      0   100%
ibrl/environments/newcomb.py                                     5      0   100%
ibrl/environments/policy_dependent_bandit.py                     8      3    62%   12, 15-16
ibrl/environments/switching.py                                  23     13    43%   16-18, 22-29, 32, 35-39
ibrl/infrabayesian/__init__.py                                   7      0   100%
ibrl/infrabayesian/a_measure.py                                 19      3    84%   35-36, 50
ibrl/infrabayesian/infradistribution.py                         69      4    94%   119-121, 124
ibrl/infrabayesian/world_model.py                               29      8    72%   23, 32, 40, 45, 56, 61, 70, 79
ibrl/infrabayesian/world_models/__init__.py                      5      0   100%
ibrl/infrabayesian/world_models/base.py                         29      8    72%   23, 32, 40, 45, 56, 61, 70, 79
ibrl/infrabayesian/world_models/bernoulli_world_model.py        65      7    89%   134-140
ibrl/infrabayesian/world_models/init.py                          5      5     0%   1-6
ibrl/infrabayesian/world_models/newcomb_world_model.py          51     31    39%   38-47, 50-51, 55-56, 59-63, 66, 73, 76, 87-99, 111-121, 131-136, 139
ibrl/infrabayesian/world_models/supra_pomdp_world_model.py     162      3    98%   11, 140, 168
ibrl/outcome.py                                                  6      0   100%
ibrl/simulators/__init__.py                                      2      0   100%
ibrl/simulators/simulator.py                                    41      4    90%   48, 50, 54, 72
ibrl/utils/__init__.py                                           3      0   100%
ibrl/utils/belief_discretization.py                             34      1    97%   25
ibrl/utils/construction.py                                      40      3    92%   34-35, 80
ibrl/utils/debug.py                                              5      3    40%   8-10
ibrl/utils/sampling.py                                           5      0   100%
------------------------------------------------------------------------------------------
TOTAL                                                         1156    251    78%
