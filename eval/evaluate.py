#!/usr/bin/env python3
"""Evaluate emitted method artifacts without importing method code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITION_TO_EXPERIMENT = {
    "random_position_speed": "randomized_air_hockey",
    "speed_wide": "velocity_range_wide",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", default=None)
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    artifact_paths = [Path(p) for p in args.artifacts] if args.artifacts else sorted(Path("outputs").glob("spiking_eprop_*.json"))
    experiments: dict[str, dict] = {}
    for path in artifact_paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        method = data.get("method", "spiking_eprop")
        experiment = CONDITION_TO_EXPERIMENT.get(data.get("condition", ""), data.get("condition", path.stem))
        experiments.setdefault(experiment, {"results": {}})
        result = {
            "success_rate": float(data["success_rate"]),
            "hit_rate": float(data.get("hit_rate", 0.0)),
        }
        if "hidden_neurons" in data:
            result["hidden_neurons"] = float(data["hidden_neurons"])
        experiments[experiment]["results"][method] = result

    scores = {"experiments": experiments}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
