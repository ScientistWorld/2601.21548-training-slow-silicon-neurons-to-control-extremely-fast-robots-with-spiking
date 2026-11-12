#!/usr/bin/env python3
"""Held-out test scoring for neuromorphic air-hockey interception controllers.

Every method directory under ``outputs/`` is scored, together with every
``proposed`` method of ``scoring/reference.json`` (so a missing submission is
scored as failed rather than skipped).  For each experiment a method must
provide ``outputs/<method>/<experiment>/seed_<k>.npz`` for k = 0..4: five
independently trained controller configurations.  The evaluator executes each
controller in closed loop on held-out episodes that exist only here, with its
own frozen copies of the task simulator and of the controller runtime, and
computes every metric from those rollouts.  Numbers written by a method are
never read.

A (method, experiment) pair fails - and receives the worst value of every
metric - when a controller is missing, unreadable, malformed, non-finite, out
of physical bounds, larger than the experiment's unit budget, or duplicated
across the five seeds.  Baselines are computed here, not read from disk.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from eval.test.chip_runtime import ChipBatch, ConfigError, controller_units, load_config
from eval.test.task_sim import AirHockeyBatch, sample_initial_conditions

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
SCORES = ROOT / "scoring" / "scores_test.json"
REFERENCE = ROOT / "scoring" / "reference.json"

# experiment -> (task condition, hidden-unit budget)
EXPERIMENTS = {
    "randomized_air_hockey": ("random_position_speed", 1020),
    "velocity_range_wide": ("speed_wide", 1020),
    "velocity_range_extreme": ("speed_extreme", 1020),
    "single_core_extreme": ("speed_extreme", 256),
}
N_CONTROLLERS = 5
N_EPISODES = 1000
BATCH = 500
EPISODE_SEED = 918_273_645_501
BASELINES = ("random_policy", "always_forward")


def worst(experiment: str) -> dict[str, float]:
    return {
        "success_rate": 0.0,
        "hit_rate": 0.0,
        "seed_worst_success": 0.0,
        "controller_units": float(EXPERIMENTS[experiment][1]),
    }


def held_out_episodes(experiment: str) -> np.ndarray:
    index = list(EXPERIMENTS).index(experiment)
    rng = np.random.default_rng([EPISODE_SEED, index])
    return sample_initial_conditions(EXPERIMENTS[experiment][0], rng, N_EPISODES)


def _run(init: np.ndarray, choose) -> tuple[np.ndarray, np.ndarray]:
    env = AirHockeyBatch(init)
    rows = np.arange(env.batch)
    obs = env.observe()
    state = choose(None, rows, None)
    while rows.size:
        actions = np.zeros(env.batch, dtype=np.int64)
        actions[rows] = choose(obs[rows], rows, state)
        obs, _, done, info = env.step(actions)
        keep = ~done[rows]
        if not np.all(keep):
            rows = rows[keep]
            if state is not None and hasattr(state, "keep"):
                state.keep(np.flatnonzero(keep))
    return info["success"].copy(), info["hit"].copy()


def rollout_controller(config: dict, init: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    succ, hit = [], []
    for start in range(0, init.shape[0], BATCH):
        part = init[start:start + BATCH]

        def choose(obs, rows, chip):
            if obs is None:
                return ChipBatch(config, part.shape[0])
            return np.argmax(chip.scores(chip.step(obs)), axis=1)

        s, h = _run(part, choose)
        succ.append(s)
        hit.append(h)
    return np.concatenate(succ), np.concatenate(hit)


def rollout_baseline(name: str, init: np.ndarray, stream: int, experiment: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([EPISODE_SEED, list(EXPERIMENTS).index(experiment), stream, 99])

    def choose(obs, rows, state):
        if obs is None:
            return None
        if name == "random_policy":
            return rng.integers(0, 2, size=rows.size)
        return np.ones(rows.size, dtype=np.int64)

    return _run(init, choose)


def _digest(config: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(config):
        h.update(key.encode())
        h.update(np.ascontiguousarray(config[key]).tobytes())
    return h.hexdigest()


def load_controllers(method_dir: Path, experiment: str) -> list[dict]:
    configs = []
    for k in range(N_CONTROLLERS):
        path = method_dir / experiment / f"seed_{k}.npz"
        if not path.is_file():
            raise ConfigError(f"missing {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        config = load_config(path)
        budget = EXPERIMENTS[experiment][1]
        if controller_units(config) > budget:
            raise ConfigError(f"seed_{k}: {controller_units(config)} hidden units exceed the "
                              f"{budget}-unit budget of {experiment}")
        configs.append(config)
    digests = [_digest(c) for c in configs]
    if len(set(digests)) != len(digests):
        raise ConfigError("controllers of different seeds are identical")
    return configs


def evaluate_method_experiment(method_dir: Path, experiment: str) -> dict:
    """Per-controller held-out outcomes for one method directory."""
    configs = load_controllers(method_dir, experiment)
    init = held_out_episodes(experiment)
    per_success, per_hit = [], []
    for config in configs:
        s, h = rollout_controller(config, init)
        per_success.append(s)
        per_hit.append(h)
    return {
        "success": np.stack(per_success),
        "hit": np.stack(per_hit),
        "units": [controller_units(c) for c in configs],
    }


def evaluate_baseline(name: str, experiment: str) -> dict:
    init = held_out_episodes(experiment)
    per_success, per_hit = [], []
    for k in range(N_CONTROLLERS):
        s, h = rollout_baseline(name, init, k, experiment)
        per_success.append(s)
        per_hit.append(h)
    return {"success": np.stack(per_success), "hit": np.stack(per_hit), "units": [0] * N_CONTROLLERS}


def summarize(outcome: dict) -> dict[str, float]:
    success = outcome["success"].astype(np.float64)
    hit = outcome["hit"].astype(np.float64)
    return {
        "success_rate": float(success.mean()),
        "hit_rate": float(hit.mean()),
        "seed_worst_success": float(success.mean(axis=1).min()),
        "controller_units": float(max(outcome["units"])),
    }


def discover_methods() -> tuple[list[str], dict[str, str]]:
    notes: dict[str, str] = {}
    names: set[str] = set()
    if OUTPUTS.is_dir():
        for path in OUTPUTS.iterdir():
            if not path.is_dir() or path.name.startswith((".", "_")):
                continue
            if path.name in BASELINES:
                notes[path.name] = "directory name is reserved for an evaluator baseline; ignored"
                continue
            names.add(path.name)
    try:
        reference = json.loads(REFERENCE.read_text())
        for payload in reference.get("experiments", {}).values():
            for name, result in payload.get("results", {}).items():
                if isinstance(result, dict) and result.get("type") == "proposed" \
                        and not name.startswith("_") and name not in BASELINES:
                    names.add(name)
    except (OSError, ValueError, AttributeError):
        pass
    return sorted(names), notes


def score_task(task: tuple[str, str, bool]) -> tuple[str, str, dict, dict]:
    """Score one (method, experiment) pair; never raises."""
    method, experiment, is_baseline = task
    start = time.time()
    if is_baseline:
        return method, experiment, summarize(evaluate_baseline(method, experiment)), {}
    try:
        outcome = evaluate_method_experiment(OUTPUTS / method, experiment)
        metrics = summarize(outcome)
        diag = {"status": "ok", "per_controller_success": [float(x) for x in outcome["success"].mean(axis=1)]}
    except ConfigError as exc:
        metrics, diag = worst(experiment), {"status": "failed", "reason": str(exc)}
    except Exception as exc:  # a submission must never crash scoring
        metrics = worst(experiment)
        diag = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3)}
    diag["seconds"] = round(time.time() - start, 1)
    return method, experiment, metrics, diag


def _workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def main() -> int:
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    if SCORES.exists():
        SCORES.unlink()
    methods, notes = discover_methods()
    tasks = [(m, e, False) for m in methods for e in EXPERIMENTS] + \
            [(b, e, True) for b in BASELINES for e in EXPERIMENTS]
    n = min(_workers(), len(tasks))
    if n > 1:
        with ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("fork")) as pool:
            done = list(pool.map(score_task, tasks))
    else:
        done = [score_task(t) for t in tasks]
    experiments: dict[str, dict] = {exp: {"results": {}} for exp in EXPERIMENTS}
    diagnostics: dict[str, dict] = {name: {"note": note} for name, note in notes.items()}
    for method, experiment, metrics, diag in done:
        experiments[experiment]["results"][method] = metrics
        if diag:
            diagnostics.setdefault(method, {})[experiment] = diag
            print(f"[test] {method}/{experiment}: {metrics} {diag.get('status')} ({diag['seconds']}s)",
                  file=sys.stderr, flush=True)
    payload = {
        "slice": "test",
        "experiments": experiments,
        "diagnostics": diagnostics,
        "protocol": {
            "controllers_per_experiment": N_CONTROLLERS,
            "episodes_per_controller": N_EPISODES,
            "unit_budget": {e: b for e, (_, b) in EXPERIMENTS.items()},
            "action_selection": "argmax of readout scores",
        },
    }
    tmp = SCORES.with_suffix(".json.partial")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(SCORES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
