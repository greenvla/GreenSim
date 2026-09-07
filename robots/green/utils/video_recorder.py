# video_recorder.py
from pathlib import Path
from typing import Optional, Dict
import cv2
import numpy as np


class VideoRecorder:
    """
    Implements FrameHandler protocol.
    Automatically starts/stops recording per episode.
    """

    def __init__(self, video_dir: str | Path = "videos", fps: float = 30.0):
        """
        Initialize the VideoRecorder.

        Creates the output directory if it doesn't exist. Video files are saved per episode
        and per camera, with filenames formatted as:
        `episode_{episode_id:04d}_{cam_name}.mp4`.

        Args:
            video_dir (str | Path): Directory where video files will be saved. Defaults to "videos".
            fps (float): Frames per second for the output video. Defaults to 30.0.
        """
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._current_episode_id: Optional[int] = None
        self._video_writers: Dict[str, cv2.VideoWriter] = {}
        self._video_paths: Dict[str, Path] = {}

    def on_episode_start(self, episode_id: int):
        """
        Callback triggered at the start of a new episode.

        Stores the current episode ID. Video writers are not initialized here;
        they are created lazily upon receiving the first frame for each camera.

        Args:
            episode_id (int): Unique identifier for the new episode.
        """
        self._current_episode_id = episode_id
        # Writers will be created lazily on first frame

    def on_frame(self, cam_name: str, rgb_image: np.ndarray, timestamp: float):
        """
        Callback for each captured RGB frame.

        If a video writer for the given camera does not exist, it is initialized.
        The RGB image is converted to BGR (as required by OpenCV) and written to the video file.

        Args:
            cam_name (str): Name of the camera that captured the frame.
            rgb_image (np.ndarray): RGB image array of shape (H, W, 3).
            timestamp (float): Simulation time at which the frame was captured (unused in recording).
        """
        if self._current_episode_id is None:
            return

        if cam_name not in self._video_writers:
            self._init_writer(cam_name, rgb_image)

        writer = self._video_writers.get(cam_name)
        if writer is not None:
            bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

    def _init_writer(self, cam_name: str, img: np.ndarray):
        """
        Initialize a VideoWriter for a specific camera.

        Constructs the output file path, sets the codec to 'mp4v', and attempts to open
        the writer. On success, stores the writer and path; on failure, logs an error
        and stores None to avoid repeated attempts.

        Args:
            cam_name (str): Name of the camera.
            img (np.ndarray): Sample image used to determine video resolution.
        """
        h, w = img.shape[:2]
        path = self.video_dir / f"episode_{self._current_episode_id:04d}_{cam_name}.mp4"
        self._video_paths[cam_name] = path

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))

        if writer.isOpened():
            self._video_writers[cam_name] = writer
            print(f"[VideoRecorder] Recording {cam_name} → {path}")
        else:
            print(f"[VideoRecorder] Failed to open writer for {path}")
            self._video_writers[cam_name] = None

    def on_episode_end(self):
        """
        Callback triggered at the end of an episode.

        Releases all active VideoWriter instances, logs the paths of successfully saved videos,
        and clears internal state to prepare for the next episode.
        """
        for cam_name, writer in self._video_writers.items():
            if writer is not None:
                writer.release()
                path = self._video_paths.get(cam_name)
                if path and path.exists():
                    print(f"[VideoRecorder] Saved: {path}")
        self._video_writers.clear()
        self._video_paths.clear()
        self._current_episode_id = None

    def close(self):
        """
        Finalize recording and release resources.

        Calls on_episode_end() to ensure any active episode is properly closed.
        Safe to call multiple times.
        """
        self.on_episode_end()