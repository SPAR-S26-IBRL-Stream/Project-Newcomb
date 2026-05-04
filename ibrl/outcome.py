from dataclasses import dataclass


@dataclass
class Outcome:
    reward: float
    observation: int | None = None
    env_action: int | None = None
