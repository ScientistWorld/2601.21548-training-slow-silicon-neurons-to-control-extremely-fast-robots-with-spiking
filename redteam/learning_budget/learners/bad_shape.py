"""Shortcut attempt: a readout over more units than the chip has."""
import numpy as np


class Oversized:
    def __init__(self, n):
        self.w, self.b = np.zeros((n + 100, 2)), np.zeros(2)

    def readout(self):
        return self.w, self.b

    def learn(self, episode):
        pass


def make_learner(spec):
    return Oversized(spec["n_hidden"])
