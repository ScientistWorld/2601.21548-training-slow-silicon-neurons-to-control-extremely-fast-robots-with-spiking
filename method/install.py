#!/usr/bin/env python3
"""Install a learner submission: ``outputs/<name>/learner.py`` + ``config.json``.

The evaluators load ``learner.py`` from each method directory and run it
online on their own controller instances; ``config.json`` holds the learner's
hyperparameters (``{"default": {...}, "<experiment>": {...overrides}}``).
Without ``--config`` the selected Eq. 1 hyperparameters are installed from
method/selected_hyperparameters.json.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "method" / "eprop_learner.py"
SELECTED = ROOT / "method" / "selected_hyperparameters.json"


def install(name, config, outputs=ROOT / "outputs", source=SOURCE):
    target = Path(outputs) / name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target / "learner.py")
    (target / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return target


def selected_config():
    selected = json.loads(SELECTED.read_text())
    return {"default": selected["default"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="spiking_eprop")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--config", help="JSON config; default: selected Eq. 1 hyperparameters")
    args = parser.parse_args()
    config = json.loads(args.config) if args.config else selected_config()
    print("installed", install(args.name, config, args.outputs), json.dumps(config))


if __name__ == "__main__":
    main()
