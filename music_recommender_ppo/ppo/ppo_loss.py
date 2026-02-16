import torch


def ppo_loss(
    old_log_probs,
    new_log_probs,
    advantages,
    values,
    returns,
    entropy,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    """
    Computes PPO loss components
    """

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )

    ratio = torch.exp(new_log_probs - old_log_probs)

    unclipped = ratio * advantages
    clipped = torch.clamp(
        ratio, 1 - clip_epsilon, 1 + clip_epsilon
    ) * advantages

    policy_loss = -torch.min(unclipped, clipped).mean()

    value_loss = (returns - values).pow(2).mean()

    entropy_loss = -entropy.mean()

    total_loss = (
        policy_loss
        + value_coef * value_loss
        + entropy_coef * entropy_loss
    )

    return total_loss, policy_loss, value_loss, entropy_loss
