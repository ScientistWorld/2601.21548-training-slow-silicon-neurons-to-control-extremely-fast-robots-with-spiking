#!/usr/bin/env python3
"""Evaluate emitted method artifacts without importing method code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARTIFACT_TO_METHOD = {
    "spiking_eprop_random_position_speed.json": "spiking_eprop",
    "random_policy_random_position_speed.json": "random_policy",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", default=["outputs/spiking_eprop_random_position_speed.json"])
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    results = {}
    for artifact in args.artifacts:
        path = Path(artifact)
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        method = ARTIFACT_TO_METHOD.get(path.name, data.get("method", path.stem))
        results[method] = {
            "success_rate": float(data["success_rate"]),
            "hit_rate": float(data.get("hit_rate", 0.0)),
            "hidden_neurons": float(data.get("hidden_neurons", 0.0)),
        }

    scores = {"experiments": {"randomized_air_hockey": {"results": results}}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
