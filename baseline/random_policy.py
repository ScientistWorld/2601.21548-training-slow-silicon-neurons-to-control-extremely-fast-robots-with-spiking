#!/usr/bin/env python3
"""Reference (non-learning) behaviours on the public task simulator.

The scored baselines ``random_policy`` (uniform random primitive each 20 ms)
and ``always_forward`` (strike immediately) are computed by the evaluators
themselves, so they cannot be altered by editing this file.  This script runs
the same behaviours on public episodes for quick inspection and writes a
summary to ``logs/baselines.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.air_hockey_env import AirHockeyBatch, sample_initial_conditions

CONDITIONS = ("random_position_speed", "speed_wide", "speed_extreme")


def run(condition: str, policy: str, episodes: int, seed: int) -> dict:
    rng = np.random.default_rng([seed, CONDITIONS.index(condition)])
    env = AirHockeyBatch(sample_initial_conditions(condition, rng, episodes))
    while not np.all(env.done):
        if policy == "random_policy":
            actions = rng.integers(0, 2, size=episodes)
        else:
            actions = np.ones(episodes, dtype=np.int64)
        env.step(actions)
    return {"success_rate": float(env.success.mean()), "hit_rate": float(env.hit.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--output", default="logs/baselines.json")
    args = parser.parse_args()
    summary = {c: {p: run(c, p, args.episodes, args.seed) for p in ("random_policy", "always_forward")}
               for c in CONDITIONS}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
