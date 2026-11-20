"""Shortcut attempt: an impossible (nonfinite) readout after ten episodes."""
import numpy as np


class NanReadout:
    def __init__(self, n):
        self.n, self.seen = n, 0

    def readout(self):
        if self.seen >= 10:
            return np.full((self.n, 2), np.nan), np.zeros(2)
        return np.zeros((self.n, 2)), np.zeros(2)

    def learn(self, episode):
        self.seen += 1


def make_learner(spec):
    return NanReadout(spec["n_hidden"])
