from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.music_env import MusicRecommendationEnv
from models.policy_network import PolicyNetwork
from models.value_network import ValueNetwork
from ppo.ppo_agent import PPOAgent


def train():
    data_dir = PROJECT_ROOT / "data" / "raw"
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    events_df = pd.read_csv(
        data_dir / "synthetic_events_logs.csv"
    )
    song_features_df = pd.read_csv(
        data_dir / "indian_songs_spotify_style_features.csv"
    )

    # Create environment
    env = MusicRecommendationEnv(
        events_df=events_df,
        song_features_df=song_features_df,
        max_steps=20,
        history_length=5,
    )

    # Infer state_dim from the initial state if observation_space is None
    initial_state = env.reset()
    state_dim = initial_state.shape[0]
    # PPOAgent is implemented for discrete action spaces only.
    if hasattr(env.action_space, "n"):
        action_dim = env.action_space.n
    else:
        raise ValueError(
            f"Unsupported action_space '{type(env.action_space).__name__}'. "
            "PPO training currently supports only discrete action spaces."
        )

    # Models
    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    # PPO Agent
    agent = PPOAgent(
        policy_net=policy_net,
        value_net=value_net,
        lr=3e-4,
        epochs=4,
    )

    num_episodes = 500

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        done = False
        while not done:
            action, log_prob, value, entropy = agent.select_action(
                state
            )

            next_state, reward, done, _ = env.step(action)

            agent.store_transition(
                state=state,
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
                entropy=entropy,
            )

            state = next_state
            episode_reward += reward

        agent.update()

        if episode % 10 == 0:
            print(
                f"Episode {episode} | "
                f"Reward: {episode_reward:.2f}"
            )

    torch.save(
        policy_net.state_dict(),
        checkpoints_dir / "policy_latest.pt",
    )
    torch.save(
        value_net.state_dict(),
        checkpoints_dir / "value_latest.pt",
    )


if __name__ == "__main__":
    train()
