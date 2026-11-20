"""Audit helper (public split only): Eq. 1 learner that stores its final readout.

Run by the visible evaluator on the public chips; afterwards each chip's
learned readout is in public_readouts/<experiment>_chip<k>.npz. Used to build
the copy shortcut below. Imports the reference rule from a sibling copy.
"""
from pathlib import Path
import numpy as np
from eprop_base import make_learner as make_eprop


def make_learner(spec):
    inner = make_eprop(spec)
    target = Path(spec["workdir"]) / "public_readouts" / f"{spec['experiment']}_chip{spec['chip']}.npz"

    class Saving:
        def __init__(self):
            self.seen = 0

        def readout(self):
            return inner.readout()

        def learn(self, episode):
            inner.learn(episode)
            self.seen += 1
            if self.seen == spec["budget"]:
                target.parent.mkdir(parents=True, exist_ok=True)
                w, b = inner.readout()
                np.savez(target, w=w, b=b)
    return Saving()
