"""Reference learner: the paper's e-prop readout rule (Eq. 1) on a fixed chip.

Paper, Sec. II-B: the fixed random spiking layer is never trained; only the
hidden-to-readout weights w_ik of the two linear readout units are plastic.
The softmax of the readout activations gives the action probabilities, and
the weights are updated every two trials with

    dw_ik = -alpha * sum_t r^t sum_{t'<=t} gamma^(t-t') (pi_k^t' - 1[a^t'=k]) sbar_i^t'

where sbar_i is the filtered spike train of hidden neuron i and r the Eq. 2
reward. Because sum_t r^t sum_{t'<=t} gamma^(t-t') x^t' = sum_t' G^t' x^t' with
the discounted return-to-go G^t' = sum_{t>=t'} gamma^(t-t') r^t, the update
below evaluates Eq. 1 exactly, summed over the two trials since the last update.

The learner sees only what the paper's host computer sees of the chip: after
each episode the filtered spike trains, the probabilities used, the sampled
actions and the rewards. alpha = learning_rate / P, where P is the mean squared
trace norm of the first two (untrained, uniformly random) trials, so one
learning_rate transfers across reservoir samples. The initial readout is zero
(uniform action probabilities); the readout biases are not plastic.

This file is self-contained (numpy only): scripts/method.sh copies it to
outputs/<method>/learner.py next to a config.json with its hyperparameters.
The switches ``critic_rate`` and ``tail_episodes`` are NOT the paper's rule;
they exist for the audit's alternative learners and are off by default.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DEFAULTS = {
    "learning_rate": 0.05,
    "gamma": 0.95,
    "update_every": 2,
    # Variant only: learned linear value readout subtracted from the return.
    "critic_rate": 0.0,
    # Variant only: freeze and sharpen the readout for the final episodes.
    "tail_episodes": 0,
    "tail_gain": 1.0,
}


def returns_to_go(reward, gamma):
    g = np.empty_like(reward)
    run = 0.0
    for t in range(len(reward) - 1, -1, -1):
        run = reward[t] + gamma * run
        g[t] = run
    return g


class EpropReadoutLearner:
    def __init__(self, n_hidden, budget, learning_rate, gamma, update_every,
                 critic_rate=0.0, tail_episodes=0, tail_gain=1.0):
        self.w = np.zeros((n_hidden, 2))
        self.b = np.zeros(2)
        self.value_w = np.zeros(n_hidden)
        self.value_b = 0.0
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.update_every = int(update_every)
        self.critic_rate = float(critic_rate)
        self.tail_start = int(budget) - int(tail_episodes)
        self.tail_gain = float(tail_gain)
        self.power = None
        self.batch = []
        self.episodes = 0

    def readout(self):
        if self.episodes >= self.tail_start:
            return self.tail_gain * self.w, self.tail_gain * self.b
        return self.w, self.b

    def learn(self, episode):
        self.episodes += 1
        if self.episodes > self.tail_start:
            return
        self.batch.append(episode)
        if len(self.batch) < self.update_every:
            return
        traces = [np.asarray(e["trace"], dtype=np.float64) for e in self.batch]
        if self.power is None:
            self.power = max(float(np.mean(np.concatenate([np.sum(s * s, axis=1) for s in traces]))), 1e-9)
        alpha = self.learning_rate / self.power
        grad = np.zeros_like(self.w)
        value_grad = np.zeros_like(self.value_w)
        value_bias_grad, n_steps = 0.0, 0
        for s, e in zip(traces, self.batch):
            p = np.asarray(e["p_forward"], dtype=np.float64)
            a = np.asarray(e["action"], dtype=np.int64)
            g = returns_to_go(np.asarray(e["reward"], dtype=np.float64), self.gamma)
            signal = g
            if self.critic_rate > 0.0:
                value = s @ self.value_w + self.value_b
                signal = g - value
                value_grad += s.T @ signal
                value_bias_grad += float(signal.sum())
            err = np.column_stack((1.0 - p, p))
            err[np.arange(len(a)), a] -= 1.0
            grad += s.T @ (signal[:, None] * err)
            n_steps += len(a)
        self.w -= alpha * grad
        if self.critic_rate > 0.0:
            # Normalised least-mean-squares step of the value readout (variant only).
            self.value_w += self.critic_rate / self.power * value_grad / n_steps
            self.value_b += self.critic_rate * value_bias_grad / n_steps
        self.batch = []


def make_learner(spec):
    config = dict(DEFAULTS)
    path = Path(spec["workdir"]) / "config.json"
    if path.exists():
        stored = json.loads(path.read_text())
        config.update(stored.get("default", {}))
        config.update(stored.get(spec["experiment"], {}))
    return EpropReadoutLearner(spec["n_hidden"], spec["budget"], **config)
