from collections import deque


class SessionAdapter:
    def __init__(
        self,
        history_length: int = 5,
        heartbeat_window_size: int = 10,
    ):
        self.history_length = history_length
        self.heartbeat_window_size = heartbeat_window_size

        self.reset()

    def reset(self):
        self.user_history = deque(maxlen=self.history_length)
        self.heartbeat_window = deque(
            maxlen=self.heartbeat_window_size
        )
        self.session_step = 0

    def update_after_recommendation(
        self,
        track_id,
        heartbeat=None,
    ):
        self.user_history.append(track_id)
        self.session_step += 1

        if heartbeat is not None:
            self.heartbeat_window.append(heartbeat)

    def get_session_context(self):
        return {
            "session_step": self.session_step,
            "user_history": self.user_history,
            "heartbeat_window": self.heartbeat_window,
        }
