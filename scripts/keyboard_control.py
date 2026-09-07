"""Non-blocking keyboard joint controller.

Reads keys from stdin and returns joint deltas each frame.
Works on Linux — uses termios + select.
"""

import io
import select
import sys
import termios
import tty
from collections import defaultdict
from typing import Dict, Tuple

# Joint index → (key, description)
# Keys are case-sensitive; lowercase = decrease, uppercase = increase.
# Step size in radians per keypress.
_STEP = 0.05
_VEL_STEP = 0.1  # m/s or rad/s per keypress for velocity commands

_KEY_MAP: Dict[str, Tuple[str, int, str]] = {
    # Torso
    "q": ("torso", 0, "torso_yaw -"),   "Q": ("torso",  0, "torso_yaw +"),
    # Left arm (torso indices 1-5)
    "a": ("torso", 1, "L_shoulder_pitch -"), "A": ("torso", 1, "L_shoulder_pitch +"),
    "s": ("torso", 2, "L_shoulder_roll -"),  "S": ("torso", 2, "L_shoulder_roll +"),
    "d": ("torso", 3, "L_shoulder_yaw -"),   "D": ("torso", 3, "L_shoulder_yaw +"),
    "f": ("torso", 4, "L_elbow_pitch -"),    "F": ("torso", 4, "L_elbow_pitch +"),
    "g": ("torso", 5, "L_elbow_yaw -"),      "G": ("torso", 5, "L_elbow_yaw +"),
    # Right arm (torso indices 6-10)
    "h": ("torso", 6,  "R_shoulder_pitch -"), "H": ("torso", 6,  "R_shoulder_pitch +"),
    "j": ("torso", 7,  "R_shoulder_roll -"),  "J": ("torso", 7,  "R_shoulder_roll +"),
    "k": ("torso", 8,  "R_shoulder_yaw -"),   "K": ("torso", 8,  "R_shoulder_yaw +"),
    "l": ("torso", 9,  "R_elbow_pitch -"),    "L": ("torso", 9,  "R_elbow_pitch +"),
    ";": ("torso", 10, "R_elbow_yaw -"),      ":": ("torso", 10, "R_elbow_yaw +"),
    # Neck (torso indices 11-12)
    "z": ("torso", 11, "neck_yaw -"),    "Z": ("torso", 11, "neck_yaw +"),
    "x": ("torso", 12, "neck_pitch -"),  "X": ("torso", 12, "neck_pitch +"),
    # Left fingers (action indices 0-5, left-first)
    "1": ("finger", 0,  "L_pinky -"),    "!": ("finger", 0,  "L_pinky +"),
    "2": ("finger", 1,  "L_ring -"),     "@": ("finger", 1,  "L_ring +"),
    "3": ("finger", 2,  "L_middle -"),   "#": ("finger", 2,  "L_middle +"),
    "4": ("finger", 3,  "L_index -"),    "$": ("finger", 3,  "L_index +"),
    "5": ("finger", 4,  "L_thumb_pitch -"), "%": ("finger", 4,  "L_thumb_pitch +"),
    "6": ("finger", 5,  "L_thumb_yaw -"),   "^": ("finger", 5,  "L_thumb_yaw +"),
    # Right fingers (action indices 6-11, right-continues)
    "7": ("finger", 6,  "R_pinky -"),    "&": ("finger", 6,  "R_pinky +"),
    "8": ("finger", 7,  "R_ring -"),     "*": ("finger", 7,  "R_ring +"),
    "9": ("finger", 8,  "R_middle -"),   "(": ("finger", 8,  "R_middle +"),
    "0": ("finger", 9,  "R_index -"),    ")": ("finger", 9,  "R_index +"),
    "-": ("finger", 10, "R_thumb_pitch -"), "_": ("finger", 10, "R_thumb_pitch +"),
    "=": ("finger", 11, "R_thumb_yaw -"),   "+": ("finger", 11, "R_thumb_yaw +"),
    # Left wrist (action: crank, roll)
    "u": ("wrist", 0, "L_wrist_crank -"), "U": ("wrist", 0, "L_wrist_crank +"),
    "i": ("wrist", 1, "L_wrist_roll -"),  "I": ("wrist", 1, "L_wrist_roll +"),
    # Right wrist (action: crank, roll)
    "o": ("wrist", 2, "R_wrist_crank -"), "O": ("wrist", 2, "R_wrist_crank +"),
    "p": ("wrist", 3, "R_wrist_roll -"),  "P": ("wrist", 3, "R_wrist_roll +"),
    # Velocity (m/s or rad/s)
    "r": ("velocity", 0, "lin_vel_x -"),   "R": ("velocity", 0, "lin_vel_x +"),
    "t": ("velocity", 1, "lin_vel_y -"),   "T": ("velocity", 1, "lin_vel_y +"),
    "v": ("velocity", 2, "lin_vel_z -"),   "V": ("velocity", 2, "lin_vel_z +"),
    "y": ("velocity", 3, "ang_vel_x -"),   "Y": ("velocity", 3, "ang_vel_x +"),
    "b": ("velocity", 4, "ang_vel_y -"),   "B": ("velocity", 4, "ang_vel_y +"),
    "n": ("velocity", 5, "ang_vel_z -"),   "N": ("velocity", 5, "ang_vel_z +"),
}


class KeyboardControl:
    """Read keyboard in a non-blocking way and accumulate joint deltas."""

    def __init__(self, step: float = _STEP):
        self._step = step
        self._old_settings = None
        self.quit_requested = False

    def start(self) -> None:
        """Set terminal to cbreak mode.

        Without a controlling terminal (docker exec without -it, nohup, cron) there is
        nothing to switch: keep the keys disabled instead of killing the run.
        """
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin)
        except (termios.error, ValueError, io.UnsupportedOperation):
            self._old_settings = None
            print("keyboard control disabled: stdin is not a terminal", flush=True)

    def stop(self) -> None:
        """Restore terminal."""
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def read_keys(self) -> list:
        """Return list of pressed key characters (non-blocking)."""
        if self._old_settings is None:      # no terminal: nothing to read
            return []
        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            keys.append(ch)
        return keys

    def get_deltas(self) -> Dict[str, defaultdict]:
        """Read keys and return accumulated joint deltas.

        Returns dict with keys: 'torso', 'finger', 'wrist', 'velocity'.
        """
        deltas = {
            "torso": defaultdict(float),
            "finger": defaultdict(float),
            "wrist": defaultdict(float),
            "velocity": defaultdict(float),
        }
        for key in self.read_keys():
            if key == "\x1b":  # ESC — request quit
                self.quit_requested = True
                self.stop()
                continue
            info = _KEY_MAP.get(key)
            if info is None:
                continue
            group, idx, desc = info
            step = _VEL_STEP if group == "velocity" else self._step
            direction = 1 if key.isupper() or key in ":!@#$%^&*()_+" else -1
            deltas[group][idx] += direction * step
        return deltas
