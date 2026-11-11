"""Helpers for hardening controls that are not the paper's method.

* ``best_delay``: per-episode enumeration of wait-then-strike delays with the
  public simulator (the brute force the test must not reward).
* ``distill_readout``: fit a readout on a fixed reservoir to the enumeration
  teacher's actions (planner distillation, out of bounds per DIRECTION.md).
* constant / malformed controller builders for shortcut cases.
"""

from __future__ import annotations

import numpy as np

from data.air_hockey_env import AirHockeyBatch, sample_initial_conditions
from data.neuromorphic_chip import ChipBatch, validate_config

MAX_DELAY = 60


def best_delay(init: np.ndarray, sim=None, chunk: int = 100) -> np.ndarray:
    """First wait (in control steps) after which striking succeeds, or -1."""
    batch_cls = AirHockeyBatch if sim is None else sim.AirHockeyBatch
    out = np.full(init.shape[0], -1, dtype=np.int64)
    delays = np.arange(MAX_DELAY)
    for start in range(0, init.shape[0], chunk):
        part = init[start:start + chunk]
        rep = np.repeat(part, MAX_DELAY, axis=0)
        wait = np.tile(delays, part.shape[0])
        env = batch_cls(rep)
        k = 0
        while not np.all(env.done):
            env.step((k >= wait).astype(np.int64))
            k += 1
        ok = env.success.reshape(part.shape[0], MAX_DELAY)
        first = np.where(ok.any(axis=1), ok.argmax(axis=1), -1)
        out[start:start + part.shape[0]] = first
    return out


def teacher_rollout(condition: str, rng: np.random.Generator, n: int):
    init = sample_initial_conditions(condition, rng, n)
    return init, best_delay(init)


def distill_readout(config: dict, condition: str, seed: int, episodes: int = 500, ridge: float = 1e-2) -> dict:
    """Behaviour-clone the enumeration teacher into the readout of ``config``."""
    rng = np.random.default_rng([seed, 4242])
    init, delay = teacher_rollout(condition, rng, episodes)
    keep = delay >= 0
    init, delay = init[keep], delay[keep]
    chip = ChipBatch(config, init.shape[0])
    env = AirHockeyBatch(init)
    rows = np.arange(init.shape[0])
    obs = env.observe()
    xs, ys = [], []
    k = 0
    while rows.size:
        trace = chip.step(obs[rows]).astype(np.float64)
        act = (k >= delay[rows]).astype(np.int64)
        xs.append(trace)
        ys.append(act)
        full = np.zeros(env.batch, dtype=np.int64)
        full[rows] = act
        obs, _, done, _ = env.step(full)
        k += 1
        alive = ~done[rows]
        if not np.all(alive):
            chip.keep(np.flatnonzero(alive))
            rows = rows[alive]
    x = np.concatenate(xs)
    y = np.concatenate(ys) * 2.0 - 1.0
    x1 = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    gram = x1.T @ x1
    lam = ridge * np.trace(gram) / gram.shape[0]
    w = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), x1.T @ y)
    out = dict(config)
    out["w_out"] = np.stack([-0.5 * w[:-1], 0.5 * w[:-1]], axis=1)
    out["b_out"] = np.array([-0.5 * w[-1], 0.5 * w[-1]])
    return validate_config(out)


def constant_controller(action: int, jitter: float) -> dict:
    """One-unit controller whose readout always prefers ``action``."""
    k = 1
    cfg = {
        "enc_centers": np.zeros((6, k)), "enc_sigma": np.ones((6, k)), "enc_max_rate_hz": np.zeros((6, k)),
        "w_in": np.zeros((1, 6 * k)), "bias": np.zeros(1), "tau_mem_ms": np.full(1, 20.0),
        "tau_syn_ms": np.full(1, 5.0), "tau_adapt_ms": np.full(1, 100.0), "adapt_increment": np.zeros(1),
        "threshold": np.ones(1), "refractory_ms": np.zeros(1), "tau_trace_ms": np.full(1, 30.0),
        "w_out": np.zeros((1, 2)), "b_out": np.zeros(2),
    }
    cfg["b_out"][action] = 1.0 + jitter
    return validate_config(cfg)
