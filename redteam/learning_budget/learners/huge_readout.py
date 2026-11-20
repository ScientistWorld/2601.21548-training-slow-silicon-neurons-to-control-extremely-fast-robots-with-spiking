"""Shortcut attempt: an out-of-range readout (1e9) that saturates every action."""
import numpy as np


def make_learner(spec):
    class Huge:
        def readout(self):
            return np.zeros((spec["n_hidden"], 2)), np.array([0.0, 1e9])

        def learn(self, episode):
            pass
    return Huge()
