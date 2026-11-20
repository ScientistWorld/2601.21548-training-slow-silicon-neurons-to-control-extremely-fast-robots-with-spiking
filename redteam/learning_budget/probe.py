"""Job-side smoke and timing probes for the learning-budget audit.

* env regression: the slot environment must reproduce the old batch
  environment's known always-forward / random success on 0.5-2.0 m/s launches
  (0.199 / 0.076 on 512 launches in the previous physics preflight);
* chip statistics and lockstep cost per 20 ms tick;
* per-tick cost of a full evaluator unit (learner processes included).
"""
from __future__ import annotations
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.air_hockey_env import AirHockeySlots, sample_initial_conditions  # noqa: E402
from data.neuromorphic_chip import ChipBank, build_chip  # noqa: E402


def env_regression():
    out = {}
    for policy in ("forward", "random"):
        rng = np.random.default_rng(7)
        n = 512
        env = AirHockeySlots(n)
        env.reset(np.arange(n), sample_initial_conditions("speed_extreme", rng, n))
        while not env.done.all():
            rows = np.flatnonzero(~env.done)
            actions = np.ones(len(rows), dtype=int) if policy == "forward" else rng.integers(0, 2, len(rows))
            env.step(rows, actions)
        out[policy] = float(env.success.mean())
    return out


def chip_timing(n_chips=20):
    start = time.time()
    chips = [build_chip("speed_extreme", [11, k]) for k in range(2)]
    build_s = (time.time() - start) / 2
    bank = ChipBank([chips[k % 2] for k in range(n_chips)])
    env = AirHockeySlots(n_chips)
    rng = np.random.default_rng(3)
    env.reset(np.arange(n_chips), sample_initial_conditions("speed_extreme", rng, n_chips))
    spikes, ticks, start = 0.0, 0, time.time()
    while not env.done.all() and ticks < 60:
        bank.step(env.obs)
        spikes += float(bank._spk.mean())
        env.step(np.flatnonzero(~env.done), rng.integers(0, 2, int((~env.done).sum())))
        ticks += 1
    seconds = time.time() - start
    return {"build_seconds_per_chip": build_s, "ticks": ticks, "ms_per_tick_chip_and_physics": 1000 * seconds / ticks,
            "mean_spike_prob_last_ms": spikes / ticks}


def unit_timing(learner_dir, n_chips=10, budget=40):
    from eval.train.evaluate import run_unit
    start = time.time()
    outcome = run_unit("probe", "randomized_air_hockey", learner_dir, n_chips, budget)
    seconds = time.time() - start
    ticks = int(outcome["steps"].astype(int).sum(axis=1).max())
    return {"chips": n_chips, "budget": budget, "seconds": seconds, "lockstep_ticks": ticks,
            "ms_per_tick": 1000 * seconds / max(1, ticks), "success": float(outcome["success"].mean()),
            "mean_episode_ticks": float(outcome["steps"].mean())}


def main():
    learner_dir = Path(sys.argv[1])
    record = {"env_regression": env_regression(), "chip_timing": chip_timing(),
              "unit_timing_10": unit_timing(learner_dir, 10), "unit_timing_20": unit_timing(learner_dir, 20)}
    print(json.dumps(record))


if __name__ == "__main__":
    main()
