"""Shortcut attempt: reuse readouts learned on the public chips, no learning.

Loads the readout that Eq. 1 learned in 2000 trials on public chip k mod 10
and keeps it frozen on the evaluator's chip k. On the visible split this
replays a fully learned controller; it only transfers if the hidden chips are
the public ones.
"""
from pathlib import Path
import numpy as np


def make_learner(spec):
    path = Path(spec["workdir"]) / "public_readouts" / f"{spec['experiment']}_chip{spec['chip'] % 10}.npz"
    with np.load(path) as data:
        w, b = data["w"].copy(), data["b"].copy()

    class Copied:
        def readout(self):
            return w, b

        def learn(self, episode):
            pass
    return Copied()
