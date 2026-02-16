import numpy as np


class ActionMapper:
    def __init__(self, song_ids):
        """
        song_ids: list of available track_ids
        """
        self.song_ids = list(song_ids)

    def action_to_track(self, action: int):
        """
        PPO outputs an integer action
        """
        action = int(action)
        action = np.clip(action, 0, len(self.song_ids) - 1)
        return self.song_ids[action]

    def action_space_size(self) -> int:
        return len(self.song_ids)
