# latest_frame_handler.py
"""Lightweight in-memory handler that stores the most recent frame per camera."""
import threading
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np

from .sensor_manager import FrameHandler


class LatestFrameHandler(FrameHandler):
    """Stores the latest RGB frame for each camera in a thread-safe dict.

    Intended for consumers (e.g. policy clients) that need the most recent
    image without shared-memory or network overhead.

    If *target_resolution* is provided, frames are resized with cv2.resize
    before storage so that downstream consumers always receive the expected
    dimensions regardless of the sensorʼs native resolution.
    """

    def __init__(
        self,
        target_resolution: Optional[Tuple[int, int]] = None,
        exclude_cameras: Optional[Set[str]] = None,
        masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self._frames: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._target_resolution = target_resolution
        self._exclude = exclude_cameras or set()
        self._masks = masks or {}

    def on_frame(self, cam_name: str, rgb_image: np.ndarray, timestamp: float) -> None:
        """Store a copy of the latest frame, resized if configured."""
        mask = self._masks.get(cam_name)
        if mask is not None:
            rgb_image = rgb_image.copy()
            rgb_image[mask == 0] = 0
        if self._target_resolution is not None and cam_name not in self._exclude:
            rgb_image = cv2.resize(rgb_image, self._target_resolution)
        with self._lock:
            self._frames[cam_name] = rgb_image.copy()

    def on_episode_start(self, episode_id: int) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    def get_latest(self, cam_name: str) -> Optional[np.ndarray]:
        """Return the latest frame for *cam_name*, or None if never captured."""
        with self._lock:
            return self._frames.get(cam_name)

    def close(self) -> None:
        with self._lock:
            self._frames.clear()
