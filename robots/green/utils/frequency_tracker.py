import time
from typing import Optional, Callable


class FrequencyTracker:
    """
    A lightweight utility to track step frequency (Hz) in any loop.

    By default, the tracker measures frequency using real wall-clock time
    via ``time.perf_counter()``. Optionally, a custom time source can be
    injected through ``time_fn`` to measure frequency in simulation time.

    Usage:
        # Real-time tracking
        tracker = FrequencyTracker(log_interval=100)
        for step in range(...):
            # ... do work ...
            tracker.step()
            if tracker.should_log():
                print(tracker.get_info())
                tracker.update_log()

        # Simulation-time tracking
        tracker = FrequencyTracker(
            log_interval=100,
            name="SimLoop",
            target_hz=30.0,
            time_fn=lambda: float(sim.current_time),
        )
        for step in range(...):
            # ... do work ...
            tracker.step()
            if tracker.should_log():
                print(tracker.get_info())
                tracker.update_log()
    """

    _COLORS = {
        "reset": "\033[0m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "bright_yellow": "\033[1;33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bold": "\033[1m",
    }

    def __init__(
        self,
        log_interval: int = 1000,
        name: str = "Loop",
        max_duration: Optional[float] = None,
        target_hz: Optional[float] = None,
        color: Optional[str] = None,
        time_fn: Optional[Callable[[], float]] = None,
    ):
        """
        Initialize the frequency tracker.

        Args:
            log_interval: How often (in steps) to allow logging.
            name: Name for logging (e.g., "Episode", "ControlLoop").
            max_duration: Optional max runtime in the selected time domain.
                If ``time_fn`` is not provided, this is real seconds.
                If ``time_fn`` is provided, this is whatever units that
                function returns (typically sim seconds).
            target_hz: Optional target frequency for comparison.
            color: Optional ANSI color name for formatted output.
            time_fn: Optional callable returning the current time as float.
                If None, uses ``time.perf_counter``.
                Can be set to something like:
                ``lambda: float(env.sim.current_time)``
                to measure sim-time frequency.
        """
        self.log_interval = log_interval
        self.name = name
        self.max_duration = max_duration
        self.target_hz = target_hz
        self.color = color
        self.time_fn = time_fn or time.perf_counter
        self.reset()

    def _colorize(self, text: str, color: str) -> str:
        """Apply ANSI color if enabled."""
        if color in self._COLORS:
            return f"{self._COLORS[color]}{text}{self._COLORS['reset']}"
        return text

    def reset(self):
        """Reset all counters and timers (call at start of episode/loop)."""
        self.start_time = self.time_fn()
        self.last_log_time = self.start_time
        self.step_count = 0
        self.last_log_step = 0

    def step(self):
        """Call once per step."""
        self.step_count += 1

    def elapsed_time(self) -> float:
        """Total elapsed time since reset in the selected time domain."""
        return self.time_fn() - self.start_time

    def should_log(self) -> bool:
        """Returns True if it's time to log based on log_interval."""
        return self.log_interval > 0 and self.step_count - self.last_log_step >= self.log_interval

    def get_instant_freq(self) -> float:
        """Instantaneous frequency over the last log interval."""
        elapsed = self.time_fn() - self.last_log_time
        steps = self.step_count - self.last_log_step
        return steps / elapsed if elapsed > 0 else 0.0

    def get_average_freq(self) -> float:
        """Average frequency since reset."""
        elapsed = self.elapsed_time()
        return self.step_count / elapsed if elapsed > 0 else 0.0

    def get_info(self) -> str:
        """Return a formatted string with frequency info."""
        inst = self.get_instant_freq()
        avg = self.get_average_freq()
        msg = f"[{self.name}] Step {self.step_count}: Inst {inst:.2f} Hz | Avg {avg:.2f} Hz"
        if self.target_hz:
            msg += f" (target: {self.target_hz:.1f} Hz)"
        if self.color is not None:
            return self._colorize(msg, self.color)
        return msg

    def update_log(self):
        """Call after logging to reset log window."""
        self.last_log_time = self.time_fn()
        self.last_log_step = self.step_count

    def is_timed_out(self) -> bool:
        """Check if max_duration has been exceeded."""
        if self.max_duration is None:
            return False
        return self.elapsed_time() > self.max_duration

    def final_report(self) -> str:
        """Get final summary after loop ends."""
        total_steps = self.step_count
        total_time = self.elapsed_time()
        avg_freq = total_steps / total_time if total_time > 0 else 0.0
        report = f"{self.name}: {total_steps} steps in {total_time:.2f}s → {avg_freq:.2f} Hz"
        if self.target_hz:
            report += f" (target: {self.target_hz:.1f} Hz)"
        return report