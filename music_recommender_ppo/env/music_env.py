import random
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.state_builder import StateBuilder
from env.reward_function import RewardFunction
from env.action_mapper import ActionMapper


class MusicRecommendationEnv(gym.Env):
    def __init__(
        self,
        events_df,
        song_features_df,
        max_steps: int = 20,
        history_length: int = 5,
    ):
        super().__init__()

        self.events_df = events_df
        self.song_features_df = song_features_df.set_index("yt_id")

        self.state_builder = StateBuilder(
            self.song_features_df,
            history_length=history_length,
        )
        self.reward_fn = RewardFunction()
        self.action_mapper = ActionMapper(self.song_features_df.index)

        self.max_steps = max_steps
        self.history_length = history_length

        self.user_history = deque(maxlen=history_length)
        self.current_step = 0

        state_dim = history_length * len(self.state_builder.feature_cols) + 3

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(
            self.action_mapper.action_space_size()
        )

    def reset(self):
        self.current_step = 0
        self.user_history.clear()

        self.context = {
            "time_of_day": random.random(),
            "device_type": random.random(),
            "location": random.random(),
        }

        return self.state_builder.build_state(
            self.user_history, self.context
        )

    def step(self, action):
        self.current_step += 1

        track_id = self.action_mapper.action_to_track(action)
        self.user_history.append(track_id)

        # Sample a synthetic event for now
        event = self.events_df.sample(1).iloc[0].to_dict()
        reward = self.reward_fn.compute_reward(event)

        next_state = self.state_builder.build_state(
            self.user_history, self.context
        )

        done = self.current_step >= self.max_steps

        info = {
            "track_id": track_id,
            "reward": reward,
        }

        return next_state, reward, done, info
