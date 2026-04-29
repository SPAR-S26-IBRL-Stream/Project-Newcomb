from dataclasses import dataclass


@dataclass
class Outcome:
    """Result of one round of agent-environment interaction.

    Attributes:
        reward:      The scalar reward.
        observation: The environment's discrete signal (e.g. the predictor's
                     prediction, or a POMDP observation index). None for
                     environments with no separate observation channel.
    """
    reward: float
    observation: int | None = None
