#!/usr/bin/env python3
"""Random-action baseline for the air-hockey task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.air_hockey_env import AirHockeyInterceptionEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="random_position_speed")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", default="outputs/random_policy_random_position_speed.json")
    args = parser.parse_args()
    success_by_seed = []
    hit_by_seed = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        env = AirHockeyInterceptionEnv(condition=args.condition, seed=5000 + seed)
        records = []
        for _ in range(args.episodes):
            env.reset()
            done = False
            info = {"success": False, "hit": False}
            while not done:
                _, _, done, info = env.step(int(rng.integers(0, 2)))
            records.append(info)
        success_by_seed.append(float(np.mean([r["success"] for r in records])))
        hit_by_seed.append(float(np.mean([r["hit"] for r in records])))
    result = {
        "method": "random_policy",
        "condition": args.condition,
        "episodes": args.episodes,
        "seeds": args.seeds,
        "success_rate": float(np.mean(success_by_seed)),
        "success_rate_iqr": float(np.subtract(*np.percentile(success_by_seed, [75, 25]))),
        "hit_rate": float(np.mean(hit_by_seed)),
        "eval_success_by_seed": success_by_seed,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
