import pandas as pd
import torch
import torch.nn.functional as F

from models.policy_network import PolicyNetwork


def offline_pretrain():
    events_df = pd.read_csv(
        "data/raw/synthetic_events_logs.csv"
    )
    song_features_df = pd.read_csv(
        "data/raw/indian_songs_spotify_style_features.csv"
    )

    song_features_df = song_features_df.set_index("yt_id")

    feature_cols = [
        "danceability",
        "energy",
        "valence_like",
        "bpm",
        "avg_loudness_db",
        "spectral_energy",
        "spectral_centroid",
        "key",
        "mode",
    ]

    action_space = list(song_features_df.index)
    action_to_idx = {
        track_id: i
        for i, track_id in enumerate(action_space)
    }

    state_dim = len(feature_cols)
    action_dim = len(action_space)

    policy_net = PolicyNetwork(state_dim, action_dim)
    optimizer = torch.optim.Adam(
        policy_net.parameters(), lr=1e-3
    )

    epochs = 5

    for epoch in range(epochs):
        total_loss = 0.0

        for _, row in events_df.iterrows():
            track_id = row["track_id"]

            if track_id not in song_features_df.index:
                continue

            state = torch.tensor(
                song_features_df.loc[track_id][feature_cols].values,
                dtype=torch.float32,
            )

            action_idx = action_to_idx.get(track_id, None)
            if action_idx is None:
                continue

            logits = policy_net(state)
            loss = F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([action_idx]),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Pretrain Epoch {epoch} | "
            f"Loss: {total_loss:.2f}"
        )

    torch.save(
        policy_net.state_dict(),
        "checkpoints/policy_pretrained.pt",
    )


if __name__ == "__main__":
    offline_pretrain()
