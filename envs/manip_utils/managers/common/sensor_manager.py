# sensor_manager.py
from typing import Optional, List, Protocol
import numpy as np
import time
from isaaclab.envs import ManagerBasedRLEnv
# from green_challenge.utils.frequency_tracker import FrequencyTracker
from .frequency_tracker import FrequencyTracker

class FrameHandler(Protocol):
    """Protocol for frame/lifecycle handlers."""
    def on_episode_start(self, episode_id: int) -> None: ...
    def on_frame(self, cam_name: str, rgb_image: np.ndarray, timestamp: float) -> None: ...
    def on_episode_end(self) -> None: ...


class CompositeHandler:
    """
    A handler that aggregates multiple FrameHandler instances and delegates
    lifecycle and frame events to all of them.

    Attributes:
        handlers (List[FrameHandler]): List of registered frame/lifecycle handlers.
    """

    def __init__(self, handlers):
        """
        Initialize the composite handler with a list of handlers.

        Args:
            handlers (List[FrameHandler]): Handlers to delegate calls to.
        """
        self.handlers = handlers

    def on_episode_start(self, episode_id):
        """
        Notify all handlers that a new episode has started.

        Args:
            episode_id (int): Unique identifier for the episode.
        """
        for h in self.handlers:
            h.on_episode_start(episode_id)

    def on_frame(self, cam_name, rgb_image, timestamp):
        """
        Dispatch a captured RGB frame to all handlers.

        Args:
            cam_name (str): Name of the camera that captured the frame.
            rgb_image (np.ndarray): RGB image as a NumPy array.
            timestamp (float): Simulation time when the frame was captured.
        """
        for h in self.handlers:
            h.on_frame(cam_name, rgb_image, timestamp)

    def on_episode_end(self):
        """Notify all handlers that the current episode has ended."""
        for h in self.handlers:
            h.on_episode_end()

    def close(self):
        """
        Close all handlers that implement a close() method.
        Useful for releasing resources like file handles or network connections.
        """
        for h in self.handlers:
            if hasattr(h, 'close'):
                h.close()


class SensorManager:
    """
    Captures frames and manages episode lifecycle.
    Delegates all I/O to registered handlers.
    """

    def __init__(
        self,
        env: "ManagerBasedRLEnv",
        capture_freq_hz: float = 30.0,
        env_index: int = 0,
        camera_names: Optional[List[str]] = None,
        handler: Optional[FrameHandler] = None,
        should_log: bool = False,
    ):
        """
        Initialize the SensorManager.

        Args:
            env (ManagerBasedRLEnv): The Isaac Lab environment instance.
            capture_freq_hz (float): Desired frame capture frequency in Hz. Default is 30.0.
            env_index (int): Index of the environment instance (for multi-env setups). Default is 0.
            camera_names (Optional[List[str]]): List of camera sensor names to capture from.
                Must be non-empty; raises ValueError if None or empty.
            handler (Optional[FrameHandler]): Handler to receive frame and lifecycle events.
                If None, no callbacks are invoked.

        Raises:
            ValueError: If camera_names is None or empty.
        """
        self.should_log = should_log
        self.env = env.unwrapped
        self.capture_freq_hz = capture_freq_hz
        self.env_index = env_index
        self.handler = handler

        self.camera_names = camera_names
        if not camera_names:
            raise ValueError("camera_names must be a non-empty list")

        cfg = self.env.cfg
        self._control_dt = cfg.decimation * cfg.sim.dt
        self._control_freq = 1.0 / self._control_dt

        # Compute capture interval in control steps
        # self._capture_every_n_steps = max(
        #     1, int(round(self._control_freq / capture_freq_hz))
        # )

        # Expose actual capture rate for external use (e.g., video FPS)
        # self.actual_capture_freq_hz = self._control_freq / self._capture_every_n_steps
        
        self._capture_interval_s = 1.0 / capture_freq_hz
        self._last_capture_time = -float('inf')
        self.actual_capture_freq_hz = capture_freq_hz  # nominal rate

        self._step_count: int = 0
        self._current_episode_id: Optional[int] = None
        
        self.freq_tracker = FrequencyTracker(
            log_interval=15,
            name="SensorManager",
            target_hz=15.0
        )

    def start_episode(self, episode_id: int):
        self._current_episode_id = episode_id
        self._step_count = 0
        
        self._last_capture_time = -float('inf')
        
        if self.handler is not None:
            self.handler.on_episode_start(episode_id)
    
    def update(self):
        """
        Process one environment step.

        Should be called once per env.step(). Captures frames at the configured frequency
        and dispatches them to the handler if enough time has elapsed since the last capture.
        Does nothing if no episode is active.
        """
        if self._current_episode_id is None:
            return

        self._step_count += 1
        current_time = (self._step_count - 1) * self._control_dt
        
        # if (self._step_count - 1) % self._capture_every_n_steps == 0:
        #     print(f"[DEBUG] Capturing at step {self._step_count - 1}")  # ADD THIS
        #     self._capture_and_dispatch(current_time)
        if (current_time - self._last_capture_time) >= self._capture_interval_s:
            self._last_capture_time = current_time
            self._capture_and_dispatch(current_time)
            
        self.freq_tracker.step()
        # Optional: log collection frequency
        if self.should_log and self.freq_tracker.should_log():
            print(self.freq_tracker.get_info())
            self.freq_tracker.update_log()

            
    def _capture_and_dispatch(self, timestamp: float):
        """
        Internal method to capture RGB frames from specified cameras and send them to the handler.

        Performs format conversion if needed:
          - Normalizes float images (assumed [0,1]) to uint8.
          - Converts RGBA to RGB using OpenCV.
          - Expands single-channel grayscale to 3-channel RGB.

        Silently skips cameras without RGB data. Errors during capture are printed but not raised.

        Args:
            timestamp (float): Simulation timestamp for the captured frame.
        """
        if self.handler is None or self.camera_names is None:
            return

        for cam_name in self.camera_names:
            try:
                data = self.env.scene.sensors[cam_name].data.output
                if "rgb" not in data:
                    continue

                rgb_tensor = data["rgb"]
                img = rgb_tensor[self.env_index].cpu().numpy() if rgb_tensor.ndim == 4 else rgb_tensor.cpu().numpy()

                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

                if img.shape[-1] == 4:
                    from cv2 import cvtColor, COLOR_RGBA2RGB
                    img = cvtColor(img, COLOR_RGBA2RGB)
                elif img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                
                self.handler.on_frame(cam_name, img, timestamp)

            except Exception as e:
                print(f"[SensorManager] Error capturing {cam_name}: {e}")
                

    def end_episode(self):
        """
        Finalize the current episode.

        Notifies the handler that the episode has ended and resets the episode ID.
        Safe to call even if no episode is active.
        """
        if self.handler is not None:
            self.handler.on_episode_end()
        self._current_episode_id = None

    def close(self):
        """
        Clean up resources.

        Calls end_episode() to ensure proper shutdown of the current episode.
        Does not close the handler directly—handlers should be closed via CompositeHandler.close()
        or manually if needed.
        """
        self.end_episode()