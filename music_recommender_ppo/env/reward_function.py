class RewardFunction:
    def __init__(
        self,
        like_reward=1.0,
        skip_penalty=-0.5,
        dislike_penalty=-2.0,
        listen_bonus_scale=0.3,
    ):
        self.like_reward = like_reward
        self.skip_penalty = skip_penalty
        self.dislike_penalty = dislike_penalty
        self.listen_bonus_scale = listen_bonus_scale

    def compute_reward(self, event: dict) -> float:
        """
        event keys expected:
        liked, unliked, do_not_recommend,
        position_ms, duration_ms
        """

        reward = 0.0

        if event.get("liked", 0) == 1:
            reward += self.like_reward

        if event.get("unliked", 0) == 1:
            reward += self.skip_penalty

        if event.get("do_not_recommend", 0) == 1:
            reward += self.dislike_penalty

        # Listen duration bonus (bounded)
        duration = max(event.get("duration_ms", 1), 1)
        position = event.get("position_ms", 0)

        listen_ratio = min(position / duration, 1.0)
        reward += self.listen_bonus_scale * listen_ratio

        return float(reward)
