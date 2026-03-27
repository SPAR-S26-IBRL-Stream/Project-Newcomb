from .. import agents
from .. import environments
from ..infrabayesian.beliefs import BernoulliBelief, GaussianBelief, NewcombLikeBelief


def parse_argument_string(string : str) -> tuple[str, dict[str, float]]:
    """
    Parse a string that optionally includes arguments

    Input syntax:
        No arguments:       <string>
        Single argument:    <string>:<arg1>=<val1>
        Multiple arguments: <string>:<arg1>=<val1>,<arg2>=<val2>,...

    All argument values should either be floats, or tuples of floats separated by ":"

    Returns a tuple consisting of:
        the base string
        a dict of all options

    Example:
        string = "base:opt1=1,opt2=2:0.5"
    ->  "base",{"opt1":1.,"opt2":(2,0.5)}
    """
    if ":" not in string:
        return string, dict()

    name,args_str = string.split(":",1)
    args_dict = dict()
    for arg in args_str.split(","):
        arg_name,arg_val = arg.split("=",1)
        if ":" not in arg_val:
            try:
                args_dict[arg_name] = float(arg_val)
            except ValueError:
                args_dict[arg_name] = arg_val
        else:
            args_dict[arg_name] = tuple(map(float, arg_val.split(":")))
    return name, args_dict


def construct_agent(string : str, options : dict[str,int], seed_offset : int = 0x01234567) -> agents.BaseAgent:
    """
    Construct agent

    Arguments:
        string:  A string specifying the agent type and potentially agent-specific options
        options: A dictionary specifying general options for agent, namely
            num_actions: Number of discrete actions
            seed:        Seed for random number generator
            verbose:     Request debugging output
        seed_offset: Default offset, such that agent and environment can safely be initialised from same seed

    Returns:
        constructed agent
    """

    agent_types = {
        "classical":     agents.QLearningAgent,
        "bayesian":      agents.BayesianAgent,
        "exp3":          agents.EXP3Agent,
        "experimental1": agents.ExperimentalAgent1,
        "experimental2": agents.ExperimentalAgent2,
        "experimental3": agents.ExperimentalAgent3,
        "infrabayesian":  agents.InfraBayesianAgent,
    }

    belief_types = {
        "bernoulli": lambda num_actions, **kw: BernoulliBelief(num_actions),
        "gaussian":  lambda num_actions, **kw: GaussianBelief(num_actions),
        "newcomb":   lambda num_actions, **kw: NewcombLikeBelief(num_actions),
    }

    name, kwargs = parse_argument_string(string)
    if name not in agent_types:
        raise RuntimeError("Invalid agent type: " + name)

    arguments = dict()
    arguments.update(options)
    arguments.update(kwargs)
    arguments.pop("num_steps", None)  # These should not be accessible to the agent
    arguments.pop("num_runs", None)
    arguments["seed"] += seed_offset

    # For infrabayesian agent, construct beliefs and optional utility from string kwargs
    if name == "infrabayesian":
        num_act = arguments["num_actions"]
        if "belief" in arguments:
            # Single belief shorthand: belief=bernoulli -> beliefs=[BernoulliBelief(...)]
            belief_name = arguments.pop("belief")
            if belief_name not in belief_types:
                raise RuntimeError("Invalid belief type: " + str(belief_name))
            arguments["beliefs"] = [belief_types[belief_name](num_actions=num_act)]
        elif "beliefs" not in arguments:
            raise RuntimeError("infrabayesian agent requires belief= or beliefs= argument")

    return agent_types[name](**arguments)


def construct_environment(string : str, options : dict[str,int], seed_offset : int = 0x89abcdef) -> environments.BaseEnvironment:
    """
    Construct environment

    Arguments:
        string:  A string specifying the environment type and potentially environment-specific options
        options: A dictionary specifying general options for environments, namely
            num_actions: Number of discrete actions
            num_steps:   Number of steps per run (for planning)
            num_runs:    Number of runs (for planning)
            seed:        Seed for random number generator
            verbose:     Request debugging output
        seed_offset: Default offset, such that agent and environment can safely be initialised from same seed

    Returns:
        constructed environment
    """

    environment_types = {
        "bandit":              environments.BanditEnvironment,
        "bernoulli-bandit":    environments.BernoulliBanditEnvironment,
        "switching":           environments.SwitchingAdversaryEnvironment,
        "newcomb":             environments.NewcombEnvironment,
        "damascus":            environments.DeathInDamascusEnvironment,
        "asymmetric-damascus": environments.AsymmetricDeathInDamascusEnvironment,
        "coordination":        environments.CoordinationGameEnvironment,
        "pdbandit":            environments.PolicyDependentBanditEnvironment,
    }

    name, kwargs = parse_argument_string(string)
    if name not in environment_types:
        raise RuntimeError("Invalid environment: " + name)

    arguments = dict()
    arguments.update(options)
    arguments.update(kwargs)
    arguments["seed"] += seed_offset
    return environment_types[name](**arguments)
