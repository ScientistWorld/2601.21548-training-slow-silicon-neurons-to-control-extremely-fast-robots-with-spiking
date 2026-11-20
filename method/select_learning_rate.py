#!/usr/bin/env python3
"""Choose the Eq. 1 learning rate on the public validation split only.

The paper does not report alpha. Every candidate runs the visible evaluator
(eval/train/evaluate.py: public chip instances, public launches, the paper's
2000-trial budget) and the candidate with the highest weighted mean of
online and final success is written to method/selected_hyperparameters.json.
Hidden test chips and launches are never used. Each candidate's scores are
kept, so an interrupted sweep resumes.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from concurrent import futures
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from method.install import install, SELECTED  # noqa: E402

GRID = (0.012, 0.025, 0.05, 0.1, 0.2)
GAMMA = 0.95


def candidate_name(prefix, rate):
    return f"{prefix}_lr{rate:g}"


def run_candidate(sweep, name, config):
    root = Path(sweep) / name
    scores = root / "scores_train.json"
    if scores.exists():
        return name, json.loads(scores.read_text())
    install(name, config, root / "outputs")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, str(ROOT / "eval" / "train" / "evaluate.py"), "--outputs", str(root / "outputs"),
                    "--scores", str(scores), "--no-baselines", "--workers", "3"], check=True, env=env, cwd=ROOT)
    return name, json.loads(scores.read_text())


def objective(scores, name):
    weights = {e: s["weight"] for e, s in json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"].items()}
    total = 0.0
    for experiment, weight in weights.items():
        row = scores["experiments"][experiment]["results"][name]
        total += weight * 0.5 * (row["online_success"] + row["final_success"])
    return total / sum(weights.values())


def sweep(prefix, grid, extra, sweep_dir, parallel):
    configs = {candidate_name(prefix, r): {"default": {"learning_rate": r, "gamma": GAMMA, **extra}} for r in grid}
    with futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        done = dict(pool.map(lambda item: run_candidate(sweep_dir, *item), configs.items()))
    table = {name: {"config": configs[name]["default"], "objective": objective(done[name], name),
                    "results": {e: done[name]["experiments"][e]["results"][name] for e in done[name]["experiments"]}}
             for name in configs}
    best = max(table, key=lambda name: table[name]["objective"])
    return best, table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grid", type=float, nargs="+", default=list(GRID))
    parser.add_argument("--sweep-dir", type=Path, default=ROOT / "tmp" / "lr_sweep")
    parser.add_argument("--parallel", type=int, default=max(1, len(os.sched_getaffinity(0)) // 3))
    args = parser.parse_args()
    best, table = sweep("eq1", args.grid, {}, args.sweep_dir, args.parallel)
    record = {"default": table[best]["config"], "selected": best,
              "criterion": "weighted mean over experiments of (online_success + final_success) / 2 on the public split",
              "candidates": table}
    SELECTED.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: round(row["objective"], 4) for name, row in table.items()}), "->", best, flush=True)


if __name__ == "__main__":
    main()
