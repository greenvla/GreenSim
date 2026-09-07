from scipy import signal
import numpy as np


class LowPassFilter:
    def __init__(
        self, order: int, cutoff_freq: int, input_dim: int, sampling_freq: int
    ):
        """N'th order Butterworth lowpass filter

        Args:
            order (int): Filter order
            cutoff_freq (int): Cutoff frequency for the lowpass filter (Hz)
            input_dim (int): Input dimension
            sampling_freq (int): Input signal sampling frequency
        """
        self.input_dim = input_dim
        self.b, self.a = signal.butter(
            order, cutoff_freq, btype="lowpass", fs=sampling_freq
        )
        self._xs = np.zeros((len(self.b), input_dim))
        self._ys = np.zeros((len(self.a) - 1, input_dim))

    def reset(self, x: np.ndarray):
        """Resets current filter value to x

        Args:
            x (np.ndarray): reset value
        """
        self._xs[:] = x.copy()
        self._ys[:] = x.copy()

    def __call__(self, x: np.ndarray):
        """_summary_

        Args:
            x (np.ndarray): Current signal value

        Returns:
            np.ndarray: Filtered input signal
        """
        self._xs[1:] = self._xs[:-1]
        self._xs[0] = x.copy()

        y = (np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)) / self.a[0]

        self._ys[1:] = self._ys[:-1]
        self._ys[0] = y.copy()

        return self._ys[0].copy()
