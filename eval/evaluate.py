#!/usr/bin/env python3
"""Evaluate emitted controller artifacts without importing method code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONDITION_TO_EXPERIMENT = {
    "random_position_speed": "randomized_air_hockey",
    "speed_wide": "velocity_range_wide",
}


def _artifact_paths(paths: list[str] | None) -> list[Path]:
    if paths:
        return [Path(p) for p in paths]
    return sorted(Path("outputs").glob("*.json"))


def _as_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _float_list(data: dict[str, Any], key: str) -> list[float] | None:
    value = data.get(key)
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            return None
        out.append(float(item))
    return out or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", default=None)
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    experiments: dict[str, dict] = {}
    for path in _artifact_paths(args.artifacts):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        method = str(data.get("method", path.stem))
        experiment = CONDITION_TO_EXPERIMENT.get(data.get("condition", ""), data.get("condition", path.stem))
        experiments.setdefault(experiment, {"results": {}})
        result = {
            "success_rate": float(data["success_rate"]),
            "hit_rate": float(data.get("hit_rate", 0.0)),
        }
        if "hidden_neurons" in data:
            result["hidden_neurons"] = float(data["hidden_neurons"])
        success_iqr = _as_float(data, "success_rate_iqr")
        if success_iqr is not None:
            result["success_rate_iqr"] = success_iqr
        per_seed_success = _float_list(data, "eval_success_by_seed")
        if per_seed_success:
            result["seed_worst_success"] = min(per_seed_success)
        experiments[experiment]["results"][method] = result

    scores = {"experiments": experiments}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
