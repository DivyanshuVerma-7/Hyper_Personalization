import random

# 🔁 Assume your PPO policy is already implemented
# Example:
# from rl.ppo_agent import PPOAgent

# MOCK song catalog (replace with real metadata / embeddings later)
SONG_CATALOG = [
    {"id": 1, "title": "Pulse Drive", "artist": "RL Waves", "tempo": 140},
    {"id": 2, "title": "Calm Signals", "artist": "Deep Context", "tempo": 70},
    {"id": 3, "title": "Heartbeat Sync", "artist": "BioGroove", "tempo": 120},
    {"id": 4, "title": "Night Runner", "artist": "Urban Tempo", "tempo": 130},
    {"id": 5, "title": "Focus Flow", "artist": "Neural State", "tempo": 90},
]


class RecommenderService:
    def __init__(self, policy=None):
        """
        policy: PPO policy instance (already trained)
        """
        self.policy = policy

    def recommend(self, state, top_k=4):
        """
        state: numerical context vector
        returns: list of song dicts
        """

        # -----------------------------------------
        # If PPO policy exists → use it
        # -----------------------------------------
        if self.policy is not None:
            action = self.policy.get_action(state)
            # action can be:
            # - song index
            # - embedding index
            # - cluster id
            # adapt mapping as needed

            selected = SONG_CATALOG[action % len(SONG_CATALOG)]
            remaining = [s for s in SONG_CATALOG if s != selected]
            random.shuffle(remaining)

            return [selected] + remaining[: top_k - 1]

        # -----------------------------------------
        # Fallback: random recommendations
        # -----------------------------------------
        return random.sample(SONG_CATALOG, top_k)
