#!/usr/bin/env python3
"""Visible train-slice scoring from emitted method artifacts.

This evaluator is intentionally artifact-only: it reads JSON files written by
the method under outputs/ and never imports or reruns method code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONDITION_TO_EXPERIMENT = {
    "random_position_speed": "randomized_air_hockey",
    "speed_wide": "velocity_range_wide",
}


def _load_reference_methods() -> set[str] | None:
    path = Path("scoring/reference.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    methods: set[str] = set()
    for experiment in data.get("experiments", {}).values():
        methods.update(experiment.get("results", {}).keys())
    return methods or None


def _artifact_paths(paths: list[str] | None) -> list[Path]:
    if paths:
        return [Path(p) for p in paths]
    return sorted(Path("outputs").glob("*.json"))


def _experiment_name(data: dict[str, Any], path: Path) -> str:
    condition = str(data.get("condition", ""))
    return CONDITION_TO_EXPERIMENT.get(condition, condition or path.stem)


def _as_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _train_success(data: dict[str, Any]) -> float:
    visible_train = _as_float(data, "train_tail_success_rate")
    if visible_train is not None:
        return visible_train
    success = _as_float(data, "success_rate")
    if success is None:
        raise KeyError("artifact is missing success_rate")
    return success


def _learning_gain(data: dict[str, Any]) -> float | None:
    tail = _as_float(data, "train_tail_success_rate")
    first = _as_float(data, "train_first_success_rate")
    if tail is None or first is None:
        return None
    return tail - first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", default=None)
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()

    registered_methods = _load_reference_methods()
    experiments: dict[str, dict[str, Any]] = {}
    for path in _artifact_paths(args.artifacts):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        method = str(data.get("method", path.stem))
        if args.artifacts is None and registered_methods is not None and method not in registered_methods:
            continue

        experiment = _experiment_name(data, path)
        result: dict[str, float] = {"success_rate": _train_success(data)}

        hidden_neurons = _as_float(data, "hidden_neurons")
        if hidden_neurons is not None:
            result["hidden_neurons"] = hidden_neurons

        gain = _learning_gain(data)
        if gain is not None:
            result["train_success_gain"] = gain

        experiments.setdefault(experiment, {"results": {}})
        experiments[experiment]["results"][method] = result

    scores = {"slice": "train", "experiments": experiments}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
