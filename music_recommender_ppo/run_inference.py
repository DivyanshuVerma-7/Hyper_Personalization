
import torch
import pandas as pd

from models.policy_network import PolicyNetwork
from inference.recommend import MusicRecommender
from env.music_env import MusicRecommendationEnv


# Load song features
song_features_df = pd.read_csv(
    "data/raw/indian_songs_spotify_style_features.csv"
)

# Load same events file used in training
events_df = pd.read_csv(
    "data/raw/first_four_users_synthetic_events_logs.csv"
)

# Rebuild environment to auto-sync dimensions
env = MusicRecommendationEnv(events_df, song_features_df)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = PolicyNetwork(state_dim, action_dim)

policy_net.load_state_dict(
    torch.load("checkpoints/policy_latest.pt")
)

recommender = MusicRecommender(
    policy_net=policy_net,
    song_features_df=song_features_df,
)

context = {
    "time_of_day": 0.6,
    "device_type": 0.3,
    "location": 0.8,
}

for step in range(5):
    result = recommender.recommend(
        context=context,
        heartbeat=85 + step * 2,
        user_familiarity=0.7,
    )
    print(f"Step {step} → {result}")
