"""Shortcut attempt: self-reported perfect metrics, no learning.

Writes perfect-looking score files into its own directory at start-up and
then never learns. The evaluator must ignore every file a method writes.
"""
import json
from pathlib import Path
import numpy as np

FAKE = {"online_success": 1.0, "final_success": 1.0}


def make_learner(spec):
    workdir = Path(spec["workdir"])
    for name in ("scores_test.json", "scores_train.json", "metrics.json"):
        try:
            (workdir / name).write_text(json.dumps({"experiments": {spec["experiment"]: {"results": {workdir.name: FAKE}}}}))
        except OSError:
            pass

    class Frozen:
        def readout(self):
            return np.zeros((spec["n_hidden"], 2)), np.zeros(2)

        def learn(self, episode):
            pass
    return Frozen()
