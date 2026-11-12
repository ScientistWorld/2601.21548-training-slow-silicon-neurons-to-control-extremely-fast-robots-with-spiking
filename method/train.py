#!/usr/bin/env python3
"""Train the paper's spiking e-prop controller and export chip configurations.

For every experiment and reservoir seed this trains one controller online and
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

from data.neuromorphic_chip import ConfigError, save_config
from method.spiking_eprop_rl import EpropConfig, train_controller

# experiment -> (task condition, hidden units of the paper's controller)
EXPERIMENTS = {
    "randomized_air_hockey": ("random_position_speed", 1020),
    "velocity_range_wide": ("speed_wide", 1020),
    "velocity_range_extreme": ("speed_extreme", 1020),
    "single_core_extreme": ("speed_extreme", 256),
}


def parse_snapshots(specs: list[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for spec in specs:
        episodes, _, directory = spec.partition("=")
        out[int(episodes)] = Path(directory)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--output-dir", default="outputs/spiking_eprop")
    parser.add_argument("--snapshot", action="append", default=[])
    parser.add_argument("--hidden-neurons", type=int, default=None,
                        help="override the experiment's unit count (default: 1020, or 256 for single_core_extreme)")
    parser.add_argument("--cells-per-dim", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=EpropConfig.learning_rate)
    parser.add_argument("--gamma", type=float, default=EpropConfig.gamma)
    parser.add_argument("--critic-learning-rate", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="retrain even if outputs exist")
    args = parser.parse_args()

    snapshots = parse_snapshots(args.snapshot)
    out_root = Path(args.output_dir)
    failures = 0
    for experiment in args.experiments:
        condition, units = EXPERIMENTS[experiment]
        cfg = EpropConfig(
            hidden_neurons=args.hidden_neurons or units,
            cells_per_dim=args.cells_per_dim,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            critic_learning_rate=args.critic_learning_rate,
        )
        for seed in args.seeds:
            final_path = out_root / experiment / f"seed_{seed}.npz"
            log_path = out_root / experiment / f"training_seed_{seed}.json"
            snap_paths = {e: d / experiment / f"seed_{seed}.npz" for e, d in snapshots.items()}
            if not args.force and log_path.exists() and all(p.exists() for p in snap_paths.values()):
                print(f"skip {experiment} seed {seed}: outputs exist", flush=True)
                continue
            start = time.time()

            def on_checkpoint(n: int, config: dict, _paths=snap_paths) -> None:
                if n in _paths:
                    save_config(_paths[n], config)

            status = "ok"
            try:
                config, history = train_controller(
                    condition, seed, args.episodes, cfg,
                    checkpoints=tuple(snap_paths), on_checkpoint=on_checkpoint,
                    log=lambda msg, _e=experiment: print(f"{_e}: {msg}", flush=True),
                )
                save_config(final_path, config)
            except ConfigError as exc:
                # Diverged readout (non-finite or out-of-range weights): no
                # valid controller exists for this seed.
                status = f"diverged: {exc}"
                history = {"online_success": ""}
                failures += 1
            online = np.array([int(c) for c in history["online_success"]])
            history.update({
                "method": out_root.name,
                "condition": condition,
                "experiment": experiment,
                "seed": seed,
                "episodes": args.episodes,
                "status": status,
                "config": {k: v for k, v in vars(cfg).items() if k != "extra"},
                "online_success_last200": float(online[-200:].mean()) if online.size else 0.0,
                "online_success_first200": float(online[:200].mean()) if online.size else 0.0,
                "wall_seconds": time.time() - start,
            })
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(history) + "\n")
            print(f"done {experiment} seed {seed} [{status}] in {time.time() - start:.0f}s "
                  f"online first200={history['online_success_first200']:.3f} "
                  f"last200={history['online_success_last200']:.3f}", flush=True)
    if failures:
        print(f"{failures} run(s) produced no valid controller", flush=True)


if __name__ == "__main__":
    main()
