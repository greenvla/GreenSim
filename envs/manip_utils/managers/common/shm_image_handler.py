# shm_ros_image_publisher_handler.py
import mmap
import struct
import numpy as np
import os
from pathlib import Path
from typing import List, Optional
from .sensor_manager import FrameHandler
# from green_challenge.utils.frequency_tracker import FrequencyTracker
from .frequency_tracker import FrequencyTracker


class ShmRosImagePublisherHandler(FrameHandler):
    """
    Zero-overhead ROS 2 image publisher using shared memory (mmap).

    This handler publishes image frames by writing them directly into shared memory segments,
    enabling zero-copy data transfer to a separate ROS 2 node that reads from the same segments.
    Camera resolution is inferred from the first received frame. The handler manages shared
    memory setup, header packing, and frame validation automatically.

    Usage:
        handler = ShmRosImagePublisherHandler(
            camera_names=["left_head_camera", "right_head_camera"],
            node_name="isaac_shm_publisher",
            shm_base_dir="~/.cache/isaac_shm",
            max_fps=30.0,
        )
    """

    def __init__(
        self,
        camera_names: Optional[List[str]] = None,
        node_name: str = "isaac_shm_publisher",
        shm_base_dir: str = os.path.expanduser("~/.cache/isaac_shm"),
        max_fps: float = 30.0,
        should_log: bool = False,
    ) -> None:
        """
        Initialize the shared memory ROS image publisher handler.

        Args:
            camera_names: List of camera names to handle. Must be non-empty.
            node_name: Name of the associated ROS 2 node (for logging or coordination).
            shm_base_dir: Base directory for shared memory files. Expanded with `os.path.expanduser`.
            max_fps: Maximum expected publishing frequency (used for internal tracking/logging).

        Raises:
            ValueError: If `camera_names` is None or empty.
        """
        self.should_log = should_log
        self.camera_names = camera_names
        if not camera_names:
            raise ValueError("camera_names must be a non-empty list")

        self.node_name = node_name
        self.shm_base_dir = Path(shm_base_dir)
        self.max_fps = max_fps

        self._header_buffers = {}
        # Create SHM dir
        self.shm_base_dir.mkdir(parents=True, exist_ok=True)

        # Per-camera state
        self._shms = {}
        self._header_sizes = {}
        self._max_image_bytes = {}

        self._episode_id = -1

        self.freq_trackers = {}
        for camera_name in self.camera_names:
            self.freq_trackers[camera_name] = FrequencyTracker(
                log_interval=15,
                name=f"ImagePublisher[{camera_name}]",
                target_hz=15.0
            )

    def _ensure_shm(self, cam_name: str, img: np.ndarray) -> None:
        """
        Lazily initialize a shared memory segment for a given camera upon first frame.

        The shared memory file is created with a header (timestamp, height, width, channels)
        followed by space for image data with 10% headroom.

        Args:
            cam_name: Name of the camera.
            img: First image frame used to infer resolution and channel count.

        Raises:
            ValueError: If `cam_name` is not in the configured `camera_names`.
        """
        if cam_name in self._shms:
            return

        if cam_name not in self.camera_names:
            raise ValueError(f"Unknown camera: {cam_name}")

        h, w = img.shape[:2]
        c = 1 if img.ndim == 2 else img.shape[2]
        img_bytes = h * w * c

        # Allow 10% headroom
        max_img_bytes = int(img_bytes * 1.1)
        header_size = 8 + 4 + 4 + 1  # double + 2*uint32 + uint8
        self._header_buffers[cam_name] = bytearray(header_size)
        shm_size = header_size + max_img_bytes

        shm_path = self.shm_base_dir / f"{cam_name}.shm"

        # Create file
        with open(shm_path, "wb") as f:
            f.write(b"\x00" * shm_size)

        # Map it. On Linux the mapping stays valid after the descriptor is
        # closed, so the file handle does not need to be kept around.
        with open(shm_path, "r+b") as shm_file:
            shm = mmap.mmap(shm_file.fileno(), shm_size)

        # Store
        self._shms[cam_name] = shm
        self._header_sizes[cam_name] = header_size
        self._max_image_bytes[cam_name] = max_img_bytes

        print(f"[ShmHandler] Initialized SHM for {cam_name}: {h}x{w}x{c} → {shm_path}")

    def on_frame(self, cam_name: str, rgb_image: np.ndarray, timestamp: float) -> None:
        """
        Process and publish a single image frame to shared memory.

        Validates camera name, ensures image is uint8 and contiguous, checks size limits,
        then writes a binary header followed by raw pixel data into the shared memory segment.

        Args:
            cam_name: Name of the camera producing the frame.
            rgb_image: Image array (HWC or HW for grayscale). Must be convertible to uint8.
            timestamp: Timestamp associated with the frame (in seconds).

        Notes:
            - Performs zero-copy write using `rgb_image.data.cast('B')`.
            - Converts float images in [0,1] to uint8 by scaling if needed.
            - Tracks publishing frequency per camera for diagnostics.
        """
        if cam_name not in self.camera_names:
            return

        # Lazy init SHM (only once)
        if cam_name not in self._shms:
            self._ensure_shm(cam_name, rgb_image)

        # Ensure uint8 (in-place if possible)
        if rgb_image.dtype != np.uint8:
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8, copy=False)
            else:
                rgb_image = rgb_image.clip(0, 255).astype(np.uint8, copy=False)

        # Ensure contiguous (critical for .data.cast)
        if not rgb_image.flags['C_CONTIGUOUS']:
            rgb_image = np.ascontiguousarray(rgb_image)

        h, w = rgb_image.shape[:2]
        c = 1 if rgb_image.ndim == 2 else rgb_image.shape[2]
        expected_size = h * w * c

        if expected_size > self._max_image_bytes[cam_name]:
            raise RuntimeError(f"Frame too large for {cam_name}")

        # ZERO-COPY: write header + image data directly
        shm = self._shms[cam_name]
        shm.seek(0)

        # Pack and write header
        header_buf = self._header_buffers[cam_name]
        struct.pack_into("=dIIb", header_buf, 0, timestamp, h, w, c)
        shm.write(header_buf)

        # Write raw image buffer (no .tobytes()!)
        shm.write(rgb_image.data.cast('B'))

        # Move freq tracking outside hot path if possible (optional)
        self.freq_trackers[cam_name].step()
        if self.should_log and self.freq_trackers[cam_name].should_log():
            print(self.freq_trackers[cam_name].get_info())
            self.freq_trackers[cam_name].update_log()

    def on_episode_start(self, episode_id: int) -> None:
        """
        Handle the start of a new episode.

        Args:
            episode_id: Unique identifier for the episode.
        """
        self._episode_id = episode_id

    def on_episode_end(self) -> None:
        """
        Handle the end of the current episode.
        """
        self._episode_id = -1

    def close(self) -> None:
        """
        Clean up all shared memory resources.

        Closes mmap regions and underlying file descriptors.
        Should be called before program termination.
        """
        # Clean up SHM
        for shm in self._shms.values():
            shm.close()