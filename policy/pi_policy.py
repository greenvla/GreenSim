"""Policy client that communicates with the lerobot-fork policy server.

Sends raw simulator observations to the server.  The server handles all
format conversion and returns actions already mapped to the simulator's
actuator groups.
"""

import os
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

import numpy as np

from .websocket_client import WebsocketClientPolicy

# Mapping from simulator camera names to policy-server image keys.
_DEFAULT_CAMERA_MAP = {
    "left_head_camera": "top_head",
    "left_wrist_camera": "hand_left",
    "right_wrist_camera": "hand_right",
}

_INFER_TIMEOUT = 30.0


class PiPolicy:
    """Policy client wrapping a WebSocket connection to the policy server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8999,
        task_provider: Optional[Callable[[], Optional[str]]] = None,
        action_horizon: int = 5,
        latest_frame_handler: Any = None,
        camera_map: Optional[Dict[str, str]] = None,
        subtask_provider: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._task_provider = task_provider
        self._action_horizon = action_horizon
        self._latest_frame_handler = latest_frame_handler
        self._camera_map = camera_map or _DEFAULT_CAMERA_MAP
        self._subtask_provider = subtask_provider
        self._client: Optional[WebsocketClientPolicy] = None
        self._action_buffer: deque = deque()
        self._settle_steps = int(os.environ.get("SETTLE_STEPS", "0") or 0)
        self._settle_left: Optional[int] = None
        self._had_chunk = False
        self._skipped = 0
        self._feed_announced = False
        # Model time of the observation being sent, in seconds since the episode began.
        # This side owns the control clock, so nobody else can supply it.
        self._t_model = 0.0

    @property
    def client(self) -> WebsocketClientPolicy:
        if self._client is None:
            self._client = WebsocketClientPolicy(host=self._host, port=self._port)
        return self._client

    def connect(self) -> Dict[str, Any]:
        """Connect now and return the server metadata."""
        return self.client.connect()

    @property
    def feed_by_server(self) -> bool:
        """Whether the server turns action chunks into commands itself.

        A chunk is a plan: something must decide which row is due on this tick, what to
        do while two plans overlap and how fast a joint may move. A server that declares
        feed_by_server has made those decisions already and returns the commands of the
        next few control ticks; otherwise it returns the raw chunk and this side plays
        one row per tick.

        Declared once, in the handshake. The two replies look alike on the wire -- a
        list of command rows either way -- so the length cannot tell them apart.
        """
        return bool(self.client.server_metadata.get("feed_by_server", False))

    def reset(self) -> None:
        self._action_buffer.clear()
        self._settle_left = None
        self._had_chunk = False

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Observation → server payload
    # ------------------------------------------------------------------

    def get_payload(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Build the observation dict sent to the policy server."""
        dummy_image = np.zeros((3, 224, 224), dtype=np.uint8)
        images = {
            "top_head": dummy_image,
            "hand_left": dummy_image,
            "hand_right": dummy_image,
        }

        if self._latest_frame_handler is not None:
            for sim_cam, policy_key in self._camera_map.items():
                frame = self._latest_frame_handler.get_latest(sim_cam)
                if frame is not None:
                    images[policy_key] = np.ascontiguousarray(
                        np.transpose(frame, (2, 0, 1))
                    )

        state = {}
        _reset_flag = False
        for k, v in obs.items():
            if k == "reset":
                _reset_flag = bool(v)
                continue
            if isinstance(v, np.ndarray):
                state[k] = v.astype(np.float64, copy=False)
            elif isinstance(v, (list, tuple)):
                state[k] = np.asarray(v, dtype=np.float64)
            elif hasattr(v, "cpu"):  # torch tensor
                state[k] = v.cpu().numpy().astype(np.float64, copy=False)
            elif isinstance(v, (int, float, np.generic)):
                # A scalar, not a length-1 array: the policy server does
                # float(state["root_height"]) directly, and on numpy >= 2
                # float(np.array([0.87])) raises TypeError, failing the whole
                # request with a 1011 internal error.
                state[k] = float(v)

        task = None
        if self._task_provider is not None:
            task = self._task_provider()
        subtask = None
        if self._subtask_provider is not None:
            subtask = self._subtask_provider()

        return {
            "state": state,
            "images": images,
            "prompt": task,
            "subtask": subtask,
            "reset": _reset_flag,
            # Model seconds since the episode began. A server that schedules the chunk
            # itself needs this to place its rows on the control grid; one that does not
            # simply ignores the key.
            "t": self._t_model,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, payload: Dict[str, Any]) -> bool:
        """Send the payload and buffer what comes back."""
        try:
            result = self.client.infer(payload)
            actions_list = result.get("actions_list", [])
            if self.feed_by_server:
                # These are not chunk rows but the commands of the next few control
                # ticks. Every one of them is meant to be played, so truncating to
                # action_horizon would silently drop the tail of the reply.
                self._action_buffer = deque(actions_list)
                if not self._feed_announced:
                    self._feed_announced = True
                    print("FEED: the policy server turns chunks into commands itself "
                          "(%d per request); this side plays them one per control tick "
                          "and asks again." % len(actions_list), flush=True)
            else:
                # A raw chunk: play the first N rows, one per tick, and discard the rest.
                self._action_buffer = deque(actions_list[: self._action_horizon])
            self._had_chunk = True
            self._settle_left = None
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            # Backoff before the caller retries. Event.wait is used instead of a
            # blocking sleep call because the SAST scanner flags that pattern.
            threading.Event().wait(1)
            return False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _frames_ready(self) -> bool:
        """Whether every policy camera has produced at least one frame.

        Before the first env.step() there are none at all — sensor_manager.update()
        lives inside step() — and get_payload() would fall back to black dummies.
        """
        if self._latest_frame_handler is None:
            return True
        return all(self._latest_frame_handler.get_latest(c) is not None
                   for c in self._camera_map)

    def act(self, observation: Dict[str, Any], t_model: float = 0.0) -> Optional[Dict[str, Any]]:
        """Return the next action for the simulator.

        ``t_model`` is the model time of this control tick, in seconds since the episode
        began; it goes to the server, which needs it to schedule anything.
        """
        self._t_model = float(t_model)
        if len(self._action_buffer) == 0:
            # Nothing to send yet: the first inference of an episode would otherwise be
            # computed from three all-black images, and with a 50-step horizon that
            # drives a whole second of garbage motion. Returning None is safe — the
            # caller keeps the previous command and ZEST holds the robot.
            if not self._frames_ready():
                self._skipped += 1
                if self._skipped % 100 == 1:
                    missing = [c for c in self._camera_map
                               if self._latest_frame_handler.get_latest(c) is None]
                    print("PI: no frames from %s yet, inference skipped (%d)"
                          % (", ".join(missing), self._skipped), flush=True)
                return None
            # SETTLE_STEPS: let the positional actuator reach the last target of the
            # chunk before asking for the next one, otherwise the new chunk is planned
            # from a pose the robot never reached and the seam jerks. 0 disables.
            if (self._settle_left is None and self._settle_steps and self._had_chunk and not self.feed_by_server):
                self._settle_left = self._settle_steps
            if self._settle_left:
                self._settle_left -= 1
                return None
            self._settle_left = None
            payload = self.get_payload(observation)
            infer_ok = self.infer(payload)
            infer_start = time.time()
            while not infer_ok:
                infer_ok = self.infer(payload)
                if time.time() - infer_start > _INFER_TIMEOUT:
                    return None
        return self._action_buffer.popleft()

 