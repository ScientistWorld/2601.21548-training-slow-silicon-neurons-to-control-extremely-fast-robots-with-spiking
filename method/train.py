#!/usr/bin/env python3
"""Train the paper's spiking e-prop controller and export chip configurations.

For every condition and reservoir seed this trains one controller online and
writes ``<output-dir>/<experiment>/seed_<k>.npz`` (the controller the
evaluators execute) plus ``training_seed_<k>.json`` (online learning curve,
diagnostic only).  ``--snapshot E=DIR`` additionally saves the controller after
``E`` training episodes under ``DIR/<experiment>/seed_<k>.npz``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data.neuromorphic_chip import save_config
from method.spiking_eprop_rl import EpropConfig, train_controller

EXPERIMENT_OF_CONDITION = {
    "random_position_speed": "randomized_air_hockey",
    "speed_wide": "velocity_range_wide",
    "speed_extreme": "velocity_range_extreme",
}


def parse_snapshots(specs: list[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for spec in specs:
        episodes, _, directory = spec.partition("=")
        out[int(episodes)] = Path(directory)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=list(EXPERIMENT_OF_CONDITION))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--output-dir", default="outputs/spiking_eprop")
    parser.add_argument("--snapshot", action="append", default=[])
    parser.add_argument("--hidden-neurons", type=int, default=1020)
    parser.add_argument("--cells-per-dim", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=EpropConfig.learning_rate)
    parser.add_argument("--gamma", type=float, default=EpropConfig.gamma)
    parser.add_argument("--critic-learning-rate", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="retrain even if outputs exist")
    args = parser.parse_args()

    snapshots = parse_snapshots(args.snapshot)
    cfg = EpropConfig(
        hidden_neurons=args.hidden_neurons,
        cells_per_dim=args.cells_per_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        critic_learning_rate=args.critic_learning_rate,
    )
    out_root = Path(args.output_dir)
    for condition in args.conditions:
        experiment = EXPERIMENT_OF_CONDITION[condition]
        for seed in args.seeds:
            final_path = out_root / experiment / f"seed_{seed}.npz"
            log_path = out_root / experiment / f"training_seed_{seed}.json"
            snap_paths = {e: d / experiment / f"seed_{seed}.npz" for e, d in snapshots.items()}
            if not args.force and final_path.exists() and log_path.exists() and all(p.exists() for p in snap_paths.values()):
                print(f"skip {experiment} seed {seed}: outputs exist", flush=True)
                continue
            start = time.time()

            def on_checkpoint(n: int, config: dict, _paths=snap_paths) -> None:
                if n in _paths:
                    save_config(_paths[n], config)

            config, history = train_controller(
                condition, seed, args.episodes, cfg,
                checkpoints=tuple(snap_paths), on_checkpoint=on_checkpoint,
                log=lambda msg: print(msg, flush=True),
            )
            save_config(final_path, config)
            online = np.array([int(c) for c in history["online_success"]])
            history.update({
                "method": out_root.name,
                "condition": condition,
                "experiment": experiment,
                "seed": seed,
                "episodes": args.episodes,
                "config": {k: v for k, v in vars(cfg).items() if k != "extra"},
                "online_success_last200": float(online[-200:].mean()) if online.size else 0.0,
                "online_success_first200": float(online[:200].mean()) if online.size else 0.0,
                "wall_seconds": time.time() - start,
            })
            log_path.write_text(json.dumps(history) + "\n")
            print(f"done {experiment} seed {seed} in {time.time() - start:.0f}s "
                  f"online first200={history['online_success_first200']:.3f} "
                  f"last200={history['online_success_last200']:.3f}", flush=True)


if __name__ == "__main__":
    main()
