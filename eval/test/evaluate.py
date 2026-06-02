#!/usr/bin/env python3
"""Held-out test-slice scoring from emitted method artifacts.

The test slice reads the same artifact interface as the train slice, but it
uses held-out evaluation summaries instead of training-tail summaries. When
scores_train.json exists, this evaluator scores the same method/experiment
pairs that appeared on train.
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


def _required_pairs(train_scores_path: Path) -> set[tuple[str, str]] | None:
    if not train_scores_path.exists():
        return None
    scores = json.loads(train_scores_path.read_text())
    pairs: set[tuple[str, str]] = set()
    for experiment, payload in scores.get("experiments", {}).items():
        if not isinstance(payload, dict):
            continue
        results = payload.get("results", {})
        if not isinstance(results, dict):
            continue
        for method in results:
            if not str(method).startswith("_"):
                pairs.add((experiment, str(method)))
    return pairs or None


def _seed_worst_success(data: dict[str, Any]) -> float | None:
    values = _float_list(data, "eval_success_by_seed")
    if values:
        return min(values)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", default=None)
    parser.add_argument("--output", default="scoring/scores_test.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    required = None if args.artifacts else _required_pairs(Path(args.train_scores))
    scored_pairs: set[tuple[str, str]] = set()
    experiments: dict[str, dict[str, Any]] = {}

    for path in _artifact_paths(args.artifacts):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        method = str(data.get("method", path.stem))
        experiment = _experiment_name(data, path)
        pair = (experiment, method)
        if required is not None and pair not in required:
            continue

        success = _as_float(data, "success_rate")
        if success is None:
            raise KeyError(f"{path} is missing success_rate")

        result: dict[str, float] = {"success_rate": success}

        hit_rate = _as_float(data, "hit_rate")
        if hit_rate is not None:
            result["hit_rate"] = hit_rate

        controller_units = _as_float(data, "controller_units")
        if controller_units is not None:
            result["controller_units"] = controller_units

        success_iqr = _as_float(data, "success_rate_iqr")
        if success_iqr is not None:
            result["success_rate_iqr"] = success_iqr

        worst_success = _seed_worst_success(data)
        if worst_success is not None:
            result["seed_worst_success"] = worst_success

        experiments.setdefault(experiment, {"results": {}})
        experiments[experiment]["results"][method] = result
        scored_pairs.add(pair)

    if required is not None:
        missing = sorted(required - scored_pairs)
        if missing:
            formatted = ", ".join(f"{exp}/{method}" for exp, method in missing)
            raise SystemExit(f"missing test artifacts for train-scored pairs: {formatted}")

    scores = {"slice": "test", "experiments": experiments}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
