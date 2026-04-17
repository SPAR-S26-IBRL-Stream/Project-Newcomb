from dataclasses import dataclass


@dataclass
class Outcome:
    """Result of one round of agent-environment interaction.

    Attributes:
        reward:     The scalar reward.
        env_action: The environment's move (e.g. the predictor's prediction).
                    None for environments with no environment move (standard bandits).
    """
    reward: float
    env_action: int | None = None
