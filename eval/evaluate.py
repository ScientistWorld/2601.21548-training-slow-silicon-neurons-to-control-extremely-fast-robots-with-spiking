#!/usr/bin/env python3
"""Default scoring entry point: visible-slice scores written to scoring/scores.json.

Runs the train-slice protocol (closed-loop rollouts of every submitted
controller on public validation episodes) and writes the result to
``scoring/scores.json``.  It never imports or reruns ``method/`` code.
"""

from __future__ import annotations

import sys
from pathlib import Path

from eval.train.evaluate import main as train_main

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    args = sys.argv[1:] or ["--output", str(root / "scoring" / "scores.json")]
    sys.exit(train_main(args))
