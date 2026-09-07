# episode_service.py
from __future__ import annotations
import torch
from enum import IntEnum

class EpisodeService:
    EVENTS = ["timeout", "success", "aborted"]
    _EVENT_TO_IDX = {name: i for i, name in enumerate(EVENTS)}

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device

        # Unified matrix for all event types [Num types, Num envs]
        self._masks = torch.zeros(
            (len(self.EVENTS), num_envs), 
            dtype=torch.bool, 
            device=device
        )

        # episode stats (per-env)
        self._completed = torch.zeros(num_envs, dtype=torch.long, device=device)   # how many completed
        self._current_id = torch.zeros(num_envs, dtype=torch.long, device=device)  # current id (0,1,2...)

    # ---------- API for GUI/external code ----------
    def request_event(self, event_key: str, env_ids: torch.Tensor | None = None):
        """Record event by string key."""
        idx = self._EVENT_TO_IDX[event_key]
        if env_ids is None:
            self._masks[idx].fill_(True)
        else:
            self._masks[idx, env_ids] = True

    def consume_event_mask(self, event_key: str) -> torch.Tensor:
        """Get and immediately reset mask."""
        idx = self._EVENT_TO_IDX[event_key]
        mask_slice = self._masks[idx]
        
        # Clone for return, then efficiently zero out the row
        out = mask_slice.clone()
        mask_slice.fill_(False)
        return out

    def get_any_event_mask(self) -> torch.Tensor:
        """mask of envs where AT LEAST ONE event of any type occurred."""
        return self._masks.any(dim=0)

    def total_completed(self) -> int:
        return int(self._completed.sum().item())

    def reached_total(self, target_total: int) -> bool:
        return self.total_completed() >= target_total

    def get_current_episode_id(self, env_id: int) -> int:
        return int(self._current_id[env_id].item())
    # ---------- hooks called by env ----------
    def on_reset_idx(self, env_ids: torch.Tensor, episode_length_buf: torch.Tensor):
        # don't count initial reset (length == 0)
        finished = env_ids[episode_length_buf[env_ids] > 0]
        if len(finished) > 0:
            self._completed[finished] += 1
            self._current_id[finished] = self._completed[finished]

    def build_episode_events(self, term: torch.Tensor, trunc: torch.Tensor):
        done = term | trunc
        if not done.any():
            return []

        ids = torch.where(done)[0].tolist()
        events = []
        for i in ids:
            events.append(
                {
                    "env_id": i,
                    "episode_id": int(self._current_id[i].item()),
                    "completed": int(self._completed[i].item()),
                    "terminated": bool(term[i].item()),
                    "truncated": bool(trunc[i].item()),
                }
            )
        return events

    def print_events(self, events: list[dict]):
        for e in events:
            print(
                f"[env {e['env_id']}] episode={e['episode_id']} "
                f"completed={e['completed']} term={e['terminated']} trunc={e['truncated']}"
            )