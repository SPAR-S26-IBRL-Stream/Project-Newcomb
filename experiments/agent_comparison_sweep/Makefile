# Options for policies
POLICY_EPSILON=epsilon=1:500:0.01,decay_type=1     # linearly decaying epsilon up to step 800
POLICY_SOFTMAX=temperature=4:500:0.05,decay_type=1 # linearly decaying temperature up to step 800

# For each agent: filename identifier and run configuration
# Different configurations of the same type agent are considered different agents here
AGENT_1 =classical.epsilon     classical:$(POLICY_EPSILON)
AGENT_2 =classical.softmax     classical:$(POLICY_SOFTMAX)
AGENT_3 =bayesian.epsilon      bayesian:$(POLICY_EPSILON)
AGENT_4 =bayesian.softmax      bayesian:$(POLICY_SOFTMAX)
AGENT_5 =exp3                  exp3
AGENT_6 =experimental1.epsilon experimental1:$(POLICY_EPSILON)
AGENT_7 =experimental1.softmax experimental1:$(POLICY_SOFTMAX)
AGENT_8 =experimental2.epsilon experimental2:$(POLICY_EPSILON)
AGENT_9 =experimental3.epsilon experimental3:$(POLICY_EPSILON)
AGENT_10=experimental3.softmax experimental3:$(POLICY_SOFTMAX)
NUM_AGENTS=10

# For each environment: filename identifier, run configuration and additional options
ENVIRONMENT_1=bandit              bandit                  --arms 10
ENVIRONMENT_2=newcomb             newcomb
ENVIRONMENT_3=damascus            damascus
ENVIRONMENT_4=asymmetric-damascus asymmetric-damascus
ENVIRONMENT_5=coordination        coordination
ENVIRONMENT_6=pdbandit            pdbandit
ENVIRONMENT_7=switching           switching:switch_at=750
NUM_ENVIRONMENTS=7

OPTIONS=--steps 1001 --runs 500

# Each plot shows one environment and a combination of agents
# Plot name, (graph file name, graph title)+
PLOT_1=classical     classical.epsilon     "ε-greedy policy" classical.softmax     "Softmax policy"
PLOT_2=bayesian      bayesian.epsilon      "ε-greedy policy" bayesian.softmax      "Softmax policy"
PLOT_3=exp3          exp3                  "Default policy"
PLOT_4=experimental1 experimental1.epsilon "ε-greedy policy" experimental1.softmax "Softmax policy"
PLOT_5=experimental2 experimental2.epsilon "ε-greedy policy"
PLOT_6=experimental3 experimental3.epsilon "ε-greedy policy" experimental3.softmax "Softmax policy"
NUM_PLOTS=6

all:
.PHONY: all

BASE_DIR=experiments/agent_comparison_sweep
OUTPUT_DIR=$(BASE_DIR)/outputs
PLOT_DIR=$(BASE_DIR)/figures

$(OUTPUT_DIR) $(PLOT_DIR): %:
	@mkdir -p $@

# Perform simulation. One target per (environment,agent)
SOURCES=$(BASE_DIR)/main.py $(wildcard ibrl/*.py) $(wildcard ibrl/*/*.py)
define CREATE_RUN_TARGET # env agent options output
$(OUTPUT_DIR)/$(strip $4).txt: $(SOURCES) | $(OUTPUT_DIR)
	uv run python -m $$(subst /,.,$$(basename $$<)) $(strip $1) $(strip $2) $(OPTIONS) $(strip $3) > $$@
endef

# Iterate over all (environment,agent) combinations
# Exclude (bandit,experimental2|experimental3), as these agents cannot handle more than 2 actions at this point
# For each remaining pair, create a target to execute the simulation and write the results to a text file
%sequence = $(if $(word ${1},${2}),$(wordlist 1,${1},${2}),$(call %sequence,${1},${2} $(words _ ${2})))
RANGE_ENVS  :=$(call %sequence,$(NUM_ENVIRONMENTS))
RANGE_AGENTS:=$(call %sequence,$(NUM_AGENTS))
RANGE_PLOTS :=$(call %sequence,$(NUM_PLOTS))
$(foreach env,$(RANGE_ENVS),$(foreach agent,$(RANGE_AGENTS), \
	$(if $(and $(filter bandit,$(word 1,$(ENVIRONMENT_$(env)))),$(filter experimental2% experimental3%,$(word 1,$(AGENT_$(agent))))),, \
		$(eval $(call CREATE_RUN_TARGET, \
			$(word 2,$(ENVIRONMENT_$(env))), \
			$(word 2,$(AGENT_$(agent))), \
			$(wordlist 3,$(words $(ENVIRONMENT_$(env))),$(ENVIRONMENT_$(env))), \
			$(word 1,$(ENVIRONMENT_$(env))).$(word 1,$(AGENT_$(agent))) \
		)) \
	) \
))


# Create plot. One target per (environment,agent)
ALL=
define CREATE_PLOT_TARGET # env agent data(file_name file_title ...)
ALL:=$(ALL) $(PLOT_DIR)/$(strip $1).$(strip $2).png
$(PLOT_DIR)/$(strip $1).$(strip $2).png: $(BASE_DIR)/plot.gp $(foreach f,$(filter-out "% %",$3),$(OUTPUT_DIR)/$(strip $1).$(f).txt) | $(PLOT_DIR)
	gnuplot -c $$< $(BASE_DIR) $(strip $1) $(strip $2) $(strip $3)
endef

$(foreach env,$(RANGE_ENVS),$(foreach plot,$(RANGE_PLOTS), \
	$(if $(and $(filter bandit,$(word 1,$(ENVIRONMENT_$(env)))),$(filter experimental2 experimental3,$(word 1,$(PLOT_$(plot))))),, \
		$(eval $(call CREATE_PLOT_TARGET, \
			$(word 1,$(ENVIRONMENT_$(env))), \
			$(word 1,$(PLOT_$(plot))), \
			$(wordlist 2,$(words $(PLOT_$(plot))),$(PLOT_$(plot))) \
		)) \
	) \
))

all: $(ALL)
# Shortcuts to create all plots involving a specific environment or agent
.PHONY: $(foreach env,$(RANGE_ENVS),$(word 1,$(ENVIRONMENT_$(env)))) $(foreach plot,$(RANGE_PLOTS),$(word 1,$(PLOT_$(plot))))
$(foreach env,$(RANGE_ENVS),$(eval $(word 1,$(ENVIRONMENT_$(env))): $(addprefix $(PLOT_DIR)/,$(filter $(word 1,$(ENVIRONMENT_$(env)))%,$(ALL:$(PLOT_DIR)/%=%)) )))
$(foreach plot,$(RANGE_PLOTS),$(eval $(word 1,$(PLOT_$(plot))): $(addsuffix .png,$(filter %$(word 1,$(PLOT_$(plot))),$(ALL:%.png=%)) )))

.PHONY: clean
clean:
	rm -rf $(OUTPUT_DIR) $(PLOT_DIR)
