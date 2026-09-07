import numpy as np


class EMAFilter:
    def __init__(self, input_dim) -> None:
        self.alpha = 0.0
        self.input_dim = input_dim
        self.prev_y = np.zeros(input_dim)

    def reset(self, x):
        self.prev_y = x

    def __call__(self, x):
        self.prev_y = self.alpha * self.prev_y + (1.0 - self.alpha) * x
        return self.prev_y
