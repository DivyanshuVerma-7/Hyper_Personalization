import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        """
        Returns action logits
        """
        return self.net(state)

    def get_action_and_logprob(self, state):
        """
        Used during training
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, dist.entropy()

    def get_deterministic_action(self, state):
        """
        Used during evaluation / inference
        """
        logits = self.forward(state)
        return torch.argmax(logits, dim=-1)
