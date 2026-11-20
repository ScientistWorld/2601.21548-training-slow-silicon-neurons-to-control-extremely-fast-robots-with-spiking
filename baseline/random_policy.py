#!/usr/bin/env python3
"""Non-learning behaviours on the public task simulator (inspection only).

The scored baselines ``random_policy`` (forward with probability 0.5 every
20 ms) and ``always_forward`` are computed by the evaluators themselves, so
editing this file cannot change them. This script runs the same behaviours on
public launches and writes ``logs/baselines.json``.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from data.air_hockey_env import AirHockeySlots, sample_initial_conditions

CONDITIONS = {"randomized_air_hockey": "random_position_speed", "velocity_range_wide": "speed_wide",
              "velocity_range_extreme": "speed_extreme"}


def run(condition, p_forward, episodes, seed):
    rng = np.random.default_rng(seed)
    env = AirHockeySlots(episodes)
    env.reset(np.arange(episodes), sample_initial_conditions(condition, rng, episodes))
    while not env.done.all():
        rows = np.flatnonzero(~env.done)
        env.step(rows, (rng.random(len(rows)) < p_forward).astype(int))
    return float(env.success.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = {e: {"random_policy": run(c, .5, args.episodes, args.seed), "always_forward": run(c, 1., args.episodes, args.seed)}
           for e, c in CONDITIONS.items()}
    Path("logs").mkdir(exist_ok=True)
    Path("logs/baselines.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
