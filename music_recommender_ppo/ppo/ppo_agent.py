import torch
import torch.optim as optim

from ppo.rollout_buffer import RolloutBuffer
from ppo.ppo_loss import ppo_loss


class PPOAgent:
    def __init__(
        self,
        policy_net,
        value_net,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=4,
        device="cpu",
    ):
        self.policy_net = policy_net.to(device)
        self.value_net = value_net.to(device)
        self.device = device

        self.optimizer = optim.Adam(
            list(policy_net.parameters())
            + list(value_net.parameters()),
            lr=lr,
        )

        self.buffer = RolloutBuffer()

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        action, log_prob, entropy = self.policy_net.get_action_and_logprob(
            state
        )
        value = self.value_net(state)

        return (
            action.item(),
            log_prob.detach(),
            value.detach(),
            entropy.detach(),
        )

    def store_transition(
        self,
        state,
        action,
        log_prob,
        reward,
        done,
        value,
        entropy,
    ):
        self.buffer.add(
            state,
            action,
            log_prob,
            reward,
            done,
            value,
            entropy,
        )

    def update(self):
        returns, advantages = (
            self.buffer.compute_returns_and_advantages(
                self.gamma, self.gae_lambda
            )
        )

        states = torch.tensor(
            self.buffer.states, dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(
            self.buffer.actions, dtype=torch.long
        ).to(self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(
            self.device
        )
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        for _ in range(self.epochs):
            logits = self.policy_net(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            values = self.value_net(states)

            loss, policy_loss, value_loss, entropy_loss = ppo_loss(
                old_log_probs=old_log_probs,
                new_log_probs=new_log_probs,
                advantages=advantages,
                values=values,
                returns=returns,
                entropy=entropy,
                clip_epsilon=self.clip_epsilon,
                entropy_coef=self.entropy_coef,
                value_coef=self.value_coef,
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.buffer.clear()
