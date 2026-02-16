import torch
import torch.nn as nn
from torch.distributions import Categorical


class RecurrentPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 128,
    ):
        super().__init__()

        self.fc = nn.Linear(state_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, lstm_hidden_dim, batch_first=True
        )

        self.policy_head = nn.Linear(lstm_hidden_dim, action_dim)
        self.value_head = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, state_seq, hidden_state=None):
        """
        state_seq: (batch, seq_len, state_dim)
        """
        x = torch.relu(self.fc(state_seq))
        lstm_out, hidden_state = self.lstm(x, hidden_state)

        return lstm_out, hidden_state

    def get_action_and_value(self, state_seq, hidden_state=None):
        """
        Used during training
        """
        lstm_out, hidden_state = self.forward(state_seq, hidden_state)

        last_out = lstm_out[:, -1, :]

        logits = self.policy_head(last_out)
        value = self.value_head(last_out).squeeze(-1)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value, hidden_state

    def get_deterministic_action(self, state_seq, hidden_state=None):
        """
        Used during inference
        """
        lstm_out, hidden_state = self.forward(state_seq, hidden_state)
        logits = self.policy_head(lstm_out[:, -1, :])
        action = torch.argmax(logits, dim=-1)

        return action, hidden_state
