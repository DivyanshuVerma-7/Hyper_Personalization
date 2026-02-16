import torch


class ExplorationController:
    def __init__(
        self,
        base_temperature: float = 1.0,
        min_temperature: float = 0.3,
        max_temperature: float = 1.5,
    ):
        self.base_temperature = base_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

    def adjust_temperature(
        self,
        session_length: int,
        user_familiarity: float = 0.5,
    ) -> float:
        """
        session_length: how long the current session is
        user_familiarity: [0,1] (new user → 0, known user → 1)
        """

        temp = self.base_temperature

        # Reduce exploration as session progresses
        temp *= max(0.7, 1.0 - session_length * 0.02)

        # Reduce exploration for familiar users
        temp *= (1.0 - 0.5 * user_familiarity)

        return float(
            max(self.min_temperature, min(temp, self.max_temperature))
        )

    def apply_temperature(self, logits, temperature: float):
        return logits / temperature
