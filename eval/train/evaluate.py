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
    "speed_narrow": "velocity_range_narrow",
    "speed_wide": "velocity_range_wide",
}


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


def _success_iqr(data: dict[str, Any]) -> float | None:
    values = _float_list(data, "train_tail_success_by_seed")
    if values:
        values = sorted(values)
        mid = len(values) // 2
        lower = values[:mid] or values
        upper = values[-mid:] if mid else values
        return _median(upper) - _median(lower)
    return _as_float(data, "success_rate_iqr")


def _seed_worst_success(data: dict[str, Any]) -> float | None:
    values = _float_list(data, "train_tail_success_by_seed")
    if values:
        return min(values)
    values = _float_list(data, "eval_success_by_seed")
    if values:
        return min(values)
    return None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", default=None)
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()

    experiments: dict[str, dict[str, Any]] = {}
    for path in _artifact_paths(args.artifacts):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        method = str(data.get("method", path.stem))

        experiment = _experiment_name(data, path)
        result: dict[str, float] = {"success_rate": _train_success(data)}

        hit_rate = _as_float(data, "hit_rate")
        if hit_rate is not None:
            result["hit_rate"] = hit_rate

        controller_units = _as_float(data, "controller_units")
        if controller_units is not None:
            result["controller_units"] = controller_units

        gain = _learning_gain(data)
        if gain is not None:
            result["train_success_gain"] = gain

        success_iqr = _success_iqr(data)
        if success_iqr is not None:
            result["success_rate_iqr"] = success_iqr

        worst_success = _seed_worst_success(data)
        if worst_success is not None:
            result["seed_worst_success"] = worst_success

        experiments.setdefault(experiment, {"results": {}})
        experiments[experiment]["results"][method] = result

    scores = {"slice": "train", "experiments": experiments}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
