"""Shortcut attempt: skip a hard experiment by crashing in it (and only it)."""
import numpy as np


class Frozen:
    def __init__(self, n):
        self.w, self.b = np.zeros((n, 2)), np.zeros(2)

    def readout(self):
        return self.w, self.b

    def learn(self, episode):
        pass


def make_learner(spec):
    if spec["experiment"] == "velocity_range_wide":
        raise RuntimeError("declining this experiment")
    return Frozen(spec["n_hidden"])
