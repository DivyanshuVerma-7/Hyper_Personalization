import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        """
        Returns scalar V(s)
        """
        return self.net(state).squeeze(-1)
