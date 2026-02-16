import numpy as np
import pandas as pd
from collections import deque


class StateBuilder:
    def __init__(
        self,
        song_features_df: pd.DataFrame,
        history_length: int = 5
    ):
        """
        song_features_df: dataframe indexed by track_id or yt_id
        history_length: number of recent tracks to include
        """
        self.feature_cols = [
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
        self.song_features_df = song_features_df.copy()
        self.history_length = history_length
        self._coerce_feature_columns()

    def _coerce_feature_columns(self) -> None:
        """
        Ensure all model feature columns are numeric so state vectors are float32.
        """
        key_to_int = {
            "C": 0,
            "C#": 1,
            "Db": 1,
            "D": 2,
            "D#": 3,
            "Eb": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "Gb": 6,
            "G": 7,
            "G#": 8,
            "Ab": 8,
            "A": 9,
            "A#": 10,
            "Bb": 10,
            "B": 11,
        }

        for col in self.feature_cols:
            if col not in self.song_features_df.columns:
                raise KeyError(
                    f"Missing required feature column '{col}' in song_features_df"
                )

            if col == "key":
                key_series = (
                    self.song_features_df[col]
                    .astype(str)
                    .str.strip()
                    .replace({"nan": np.nan, "None": np.nan})
                )
                self.song_features_df[col] = key_series.map(key_to_int)
            else:
                self.song_features_df[col] = pd.to_numeric(
                    self.song_features_df[col], errors="coerce"
                )

            self.song_features_df[col] = (
                self.song_features_df[col]
                .fillna(0.0)
                .astype(np.float32)
            )

    def build_state(
        self,
        user_history,
        context,
        heartbeat_window=None,
    ):
        import numpy as np

        song_vectors = []

        for track_id in list(user_history)[-self.history_length:]:
            if track_id in self.song_features_df.index:
                song_vectors.append(
                    self.song_features_df.loc[track_id][
                        self.feature_cols
                    ].values
                )
            else:
                song_vectors.append(
                    np.zeros(len(self.feature_cols))
                )

        while len(song_vectors) < self.history_length:
            song_vectors.insert(
                0, np.zeros(len(self.feature_cols))
            )

        song_state = np.concatenate(song_vectors)

        # Heartbeat features
        if heartbeat_window is not None and len(heartbeat_window) > 0:
            hb_values = np.array(list(heartbeat_window))
            hb_mean = np.mean(hb_values)
            hb_std = np.std(hb_values)
            hb_trend = hb_values[-1] - hb_values[0]
        else:
            hb_mean, hb_std, hb_trend = 0.0, 0.0, 0.0

        heartbeat_vector = np.array(
            [hb_mean, hb_std, hb_trend]
        )

        context_vector = np.array([
            context.get("time_of_day", 0.0),
            context.get("device_type", 0.0),
            context.get("location", 0.0),
        ])

        return np.concatenate(
            [song_state, heartbeat_vector, context_vector]
        ).astype(np.float32)
