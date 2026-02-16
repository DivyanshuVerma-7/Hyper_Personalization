import torch


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.entropies = []

    def add(
        self,
        state,
        action,
        log_prob,
        reward,
        done,
        value,
        entropy,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.entropies.append(entropy)

    def compute_returns_and_advantages(
        self,
        gamma=0.99,
        gae_lambda=0.95,
    ):
        """
        Generalized Advantage Estimation (GAE)
        """
        returns = []
        advantages = []

        gae = 0.0
        next_value = 0.0

        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + gamma * next_value * (1 - self.dones[step])
                - self.values[step]
            )

            gae = (
                delta
                + gamma * gae_lambda * (1 - self.dones[step]) * gae
            )

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])

            next_value = self.values[step]

        return (
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
        )
