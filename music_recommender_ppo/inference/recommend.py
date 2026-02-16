import torch
import numpy as np
from torch.distributions import Categorical
from typing import Optional

from env.state_builder import StateBuilder
from env.action_mapper import ActionMapper
from inference.session_adapter import SessionAdapter
from inference.exploration_controller import ExplorationController


class MusicRecommender:
    def __init__(
        self,
        policy_net,
        song_features_df,
        device="cpu",
    ):
        self.device = device
        self.policy_net = policy_net.to(device)
        self.policy_net.eval()

        self.song_features_df = song_features_df.set_index("yt_id")

        self.state_builder = StateBuilder(
            song_features_df=self.song_features_df
        )
        self.action_mapper = ActionMapper(
            self.song_features_df.index
        )

        self.session_adapter = SessionAdapter()
        self.exploration_controller = ExplorationController()

    @torch.no_grad()
    def recommend(
        self,
        context: dict,
        heartbeat: Optional[float] = None,
        user_familiarity: float = 0.5,
    ):
        session_ctx = self.session_adapter.get_session_context()

        state = self.state_builder.build_state(
            user_history=session_ctx["user_history"],
            context=context,
            heartbeat_window=session_ctx["heartbeat_window"],
        )

        state = torch.tensor(
            state, dtype=torch.float32
        ).to(self.device)

        logits = self.policy_net(state)

        temperature = self.exploration_controller.adjust_temperature(
            session_length=session_ctx["session_step"],
            user_familiarity=user_familiarity,
        )

        logits = self.exploration_controller.apply_temperature(
            logits, temperature
        )

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        track_id = self.action_mapper.action_to_track(
            action.item()
        )

        self.session_adapter.update_after_recommendation(
            track_id, heartbeat
        )

        return {
            "track_id": track_id,
            "exploration_temperature": temperature,
            "action_probability": float(
                dist.probs[action].cpu().item()
            ),
        }
