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


# --------------------------------------------------------------------------- conjunctive reservoir
# Honest representation variant: instead of random sparse input synapses, every
# hidden unit is a coincidence detector for one (lateral puck position, lateral
# puck velocity) pair, plus a few units tuned to the end-effector position.  The
# readout is learned online with the paper's Eq. 1 exactly as the reference.

def build_conjunctive(seed: int, condition: str, units: int) -> tuple[dict, float]:
    from data.air_hockey_env import CONDITIONS
    rng = np.random.default_rng([seed, 99991])
    k = 32 if units >= 1000 else 16
    ky, kv = k, k - 1
    vmax = max(0.2, float(CONDITIONS[condition]["speed"][1]))
    centers = np.zeros((6, k))
    sigma = np.ones((6, k))
    rates = np.zeros((6, k))
    for dim, lo, hi, n in ((1, -0.5, 0.5, ky), (3, -vmax, vmax, kv), (4, 0.7, 1.5, k)):
        centers[dim, :n] = np.linspace(lo, hi, n)
        sigma[dim, :] = (hi - lo) / (n - 1)
        rates[dim, :n] = 200.0
    tau_mem, tau_syn = 20.0, 5.0
    gain = 0.2 / ((1 - np.exp(-1 / tau_syn)) * (1 - np.exp(-1 / tau_mem)))  # V per unit weight at peak rate
    w_pair, w_single = 0.62 / gain, 1.3 / gain
    w_in = np.zeros((units, 6 * k))
    i = 0
    for a in range(ky):
        for b in range(kv):
            if i >= units:
                break
            w_in[i, 1 * k + a] = w_pair
            w_in[i, 3 * k + b] = w_pair
            i += 1
    c = 0
    while i < units:
        w_in[i, 4 * k + (c % k)] = w_single
        i += 1
        c += 1
    w_in *= rng.lognormal(0.0, 0.03, w_in.shape)
    config = {
        "enc_centers": centers, "enc_sigma": sigma, "enc_max_rate_hz": rates, "w_in": w_in,
        "bias": np.zeros(units), "tau_mem_ms": np.full(units, tau_mem), "tau_syn_ms": np.full(units, tau_syn),
        "tau_adapt_ms": np.full(units, 100.0), "adapt_increment": np.zeros(units), "threshold": np.ones(units),
        "refractory_ms": np.full(units, 2.0), "tau_trace_ms": np.full(units, 30.0),
        "w_out": np.zeros((units, 2)), "b_out": np.zeros(2),
    }
    config = validate_config(config)
    calib = np.random.default_rng([seed, 104729])
    n = 16
    chip = ChipBatch(config, n)
    env = AirHockeyBatch(sample_initial_conditions(condition, calib, n))
    rows = np.arange(n)
    obs = env.observe()
    power, count = 0.0, 0
    while rows.size:
        trace = chip.step(obs[rows]).astype(np.float64)
        power += float(np.sum(trace * trace))
        count += rows.size
        act = np.zeros(n, dtype=np.int64)
        act[rows] = calib.integers(0, 2, size=rows.size)
        obs, _, d, _ = env.step(act)
        keep = ~d[rows]
        if not np.all(keep):
            chip.keep(np.flatnonzero(keep))
            rows = rows[keep]
    power = max(power / max(1, count), 1e-3)
    config["w_out"] = rng.normal(0.0, 0.1 / np.sqrt(power), (units, 2)).astype(np.float32)
    return config, power


def train_conjunctive(experiment: str, seed: int, episodes: int, lr: float, out_dir: str) -> None:
    import json
    import time
    from pathlib import Path

    from data.neuromorphic_chip import save_config
    from method.spiking_eprop_rl import EpropConfig, EpropReadoutLearner
    from method.train import EXPERIMENTS as PLAN

    condition, units = PLAN[experiment]
    out = Path(out_dir) / experiment
    if (out / f"training_seed_{seed}.json").exists():
        return
    start = time.time()
    config, power = build_conjunctive(seed, condition, units)
    learner = EpropReadoutLearner(config, power, EpropConfig(hidden_neurons=units, learning_rate=lr), seed)
    env_rng = np.random.default_rng([seed, 2601, 21548])
    succ = []
    while len(succ) < episodes:
        traj, info = learner.run_trials(sample_initial_conditions(condition, env_rng, 2))
        learner.update(traj)
        succ.extend(int(x) for x in info["success"])
    status = "ok"
    try:
        save_config(out / f"seed_{seed}.npz", learner.export())
    except Exception as exc:  # diverged readout
        status = f"failed: {exc}"
    o = np.array(succ)
    (out / f"training_seed_{seed}.json").parent.mkdir(parents=True, exist_ok=True)
    (out / f"training_seed_{seed}.json").write_text(json.dumps({
        "experiment": experiment, "seed": seed, "episodes": episodes, "learning_rate": lr, "status": status,
        "trace_power": power, "online_success": "".join(map(str, succ)),
        "online_success_first200": float(o[:200].mean()), "online_success_last200": float(o[-200:].mean()),
        "wall_seconds": time.time() - start}) + "\n")
    print(f"conjunctive {experiment} seed {seed} [{status}] last200={o[-200:].mean():.3f} "
          f"in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, "/home/user")
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    a = parser.parse_args()
    train_conjunctive(a.experiment, a.seed, a.episodes, a.learning_rate, a.output_dir)
