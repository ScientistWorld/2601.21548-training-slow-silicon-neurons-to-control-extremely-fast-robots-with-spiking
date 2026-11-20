"""Trace-ignoring control: Eq. 1 applied to the readout bias alone.

The policy cannot depend on the chip's spike trains, so it can only learn a
constant per-tick forward probability. Measures what a state-blind schedule
earns under the gym's interface.
"""
import numpy as np


def returns_to_go(reward, gamma):
    g, run = np.empty_like(reward), 0.0
    for t in range(len(reward) - 1, -1, -1):
        run = reward[t] + gamma * run
        g[t] = run
    return g


class BiasOnly:
    def __init__(self, n, rate=0.01, gamma=0.95):
        self.n, self.b, self.rate, self.gamma, self.batch = n, np.zeros(2), rate, gamma, []

    def readout(self):
        return np.zeros((self.n, 2)), self.b

    def learn(self, episode):
        self.batch.append(episode)
        if len(self.batch) < 2:
            return
        grad = np.zeros(2)
        for e in self.batch:
            p, a = e["p_forward"], e["action"]
            g = returns_to_go(e["reward"], self.gamma)
            err = np.column_stack((1.0 - p, p))
            err[np.arange(len(a)), a] -= 1.0
            grad += (g[:, None] * err).sum(axis=0)
        self.b -= self.rate * grad
        self.b = np.clip(self.b, -50, 50)
        self.batch = []


def make_learner(spec):
    return BiasOnly(spec["n_hidden"])
