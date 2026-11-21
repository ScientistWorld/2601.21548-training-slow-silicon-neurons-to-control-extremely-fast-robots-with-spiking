"""Alternative readout learners for the headroom check (not the paper's rule).

Both keep the reference's structure: the same REINFORCE-style signal as Eq. 1
(discounted return-to-go times (pi - onehot(a)) times the readout input),
updated every two trials, exporting a linear readout (w, b) of the chip's
filtered spike trains. They change how the update is conditioned:

* ``standardized``: after ``warmup`` uniformly random episodes the learner
  fixes per-neuron means and scales of the spike trains and learns on the
  standardized inputs plus an explicit bias input. Eq. 1 on raw, non-negative
  traces moves mostly along the common-mode (mean-trace) direction, which acts
  like a large global bias and can saturate the softmax; there the gradient
  vanishes and the chip never recovers. Standardizing decouples that
  direction. The exported readout folds the standardization back in:
  w = w_z / sigma, b = c - (mu / sigma) . w_z.
* ``trust``: Eq. 1 on raw traces, but each update is scaled down so that no
  visited state's logit difference changes by more than ``max_logit_change``.

Self-contained (numpy only); hyperparameters come from ``config.json``
(``{"default": {...}, "<experiment>": {...}}``).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def returns_to_go(reward, gamma):
    g, run = np.empty_like(reward), 0.0
    for t in range(len(reward) - 1, -1, -1):
        run = reward[t] + gamma * run
        g[t] = run
    return g


def policy_error(episode):
    p = np.asarray(episode["p_forward"], dtype=np.float64)
    a = np.asarray(episode["action"], dtype=np.int64)
    err = np.column_stack((1.0 - p, p))
    err[np.arange(len(a)), a] -= 1.0
    return err


class Variant:
    def __init__(self, n_hidden, mode, learning_rate, gamma=0.95, update_every=2, warmup=10,
                 max_logit_change=1.0, sigma_floor=0.1):
        self.n, self.mode = n_hidden, mode
        self.learning_rate, self.gamma, self.update_every = float(learning_rate), float(gamma), int(update_every)
        self.warmup, self.max_logit_change, self.sigma_floor = int(warmup), float(max_logit_change), float(sigma_floor)
        self.w = np.zeros((n_hidden, 2))
        self.c = np.zeros(2)
        self.mu = None
        self.inv_sigma = None
        self.power = None
        self.warm, self.batch = [], []

    # ------------------------------------------------------------------ readout
    def readout(self):
        if self.mode == "standardized":
            if self.mu is None:
                return np.zeros((self.n, 2)), np.zeros(2)
            return self.w * self.inv_sigma[:, None], self.c - (self.mu * self.inv_sigma) @ self.w
        return self.w, np.zeros(2)

    def features(self, trace):
        s = np.asarray(trace, dtype=np.float64)
        if self.mode == "standardized":
            return (s - self.mu) * self.inv_sigma
        return s

    # ------------------------------------------------------------------ learning
    def learn(self, episode):
        if self.mode == "standardized" and self.mu is None:
            self.warm.append(np.asarray(episode["trace"], dtype=np.float64))
            if len(self.warm) >= self.warmup:
                s = np.concatenate(self.warm)
                self.mu, sd = s.mean(axis=0), s.std(axis=0)
                positive = sd[sd > 0]
                floor = self.sigma_floor * (np.median(positive) if positive.size else 1.0)
                self.inv_sigma = np.where(sd > 0, 1.0 / np.maximum(sd, floor), 0.0)
                z = (s - self.mu) * self.inv_sigma
                self.power = float(np.mean(np.sum(z * z, axis=1))) + 1.0
                self.warm = []
            return
        self.batch.append(episode)
        if len(self.batch) < self.update_every:
            return
        feats = [self.features(e["trace"]) for e in self.batch]
        if self.power is None:
            self.power = max(float(np.mean(np.concatenate([np.sum(f * f, axis=1) for f in feats]))), 1e-9)
        alpha = self.learning_rate / self.power
        grad_w, grad_c = np.zeros_like(self.w), np.zeros(2)
        for f, e in zip(feats, self.batch):
            g = returns_to_go(np.asarray(e["reward"], dtype=np.float64), self.gamma)
            signal = g[:, None] * policy_error(e)
            grad_w += f.T @ signal
            grad_c += signal.sum(axis=0)
        delta_w = -alpha * grad_w
        delta_c = -alpha * grad_c if self.mode == "standardized" else np.zeros(2)
        if self.mode == "trust":
            change = np.abs(np.concatenate(feats) @ (delta_w[:, 1] - delta_w[:, 0]))
            largest = float(change.max()) if change.size else 0.0
            if largest > self.max_logit_change:
                delta_w *= self.max_logit_change / largest
        self.w += delta_w
        self.c += delta_c
        self.batch = []


def make_learner(spec):
    config = {}
    path = Path(spec["workdir"]) / "config.json"
    if path.exists():
        stored = json.loads(path.read_text())
        config.update(stored.get("default", {}))
        config.update(stored.get(spec["experiment"], {}))
    return Variant(spec["n_hidden"], **config)
