"""Weak control: never changes its zero readout (uniformly random actions)."""
import numpy as np


class Frozen:
    def __init__(self, n):
        self.w, self.b = np.zeros((n, 2)), np.zeros(2)

    def readout(self):
        return self.w, self.b

    def learn(self, episode):
        pass


def make_learner(spec):
    return Frozen(spec["n_hidden"])
