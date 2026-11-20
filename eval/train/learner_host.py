"""Process host for one submitted learner (visible validation split).

Started by eval/train/evaluate.py, one process per controller instance. It
loads ``<method>/learner.py``, calls ``make_learner(spec)``, and relays a
fixed binary protocol over stdin/stdout:

* host -> evaluator: the readout ``w`` [N, 2] then ``b`` [2], little-endian
  float64, once at start and once after every episode;
* evaluator -> host: int32 ``T`` (0 ends the session), then the episode's
  filtered spike trains [T, N] float32, forward probabilities [T] float64,
  actions [T] int8 and Eq. 2 rewards [T] float64.

The evaluator validates every readout it receives; nothing else crosses back.
Anything the learner prints goes to stderr.
"""
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path


def read_exact(fd, n):
    parts, got = [], 0
    while got < n:
        chunk = os.read(fd, n - got)
        if not chunk:
            raise SystemExit(0)
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def write_all(fd, data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def main():
    learner_file = Path(sys.argv[1]).resolve()
    spec = json.loads(sys.argv[2])
    out_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    import numpy as np
    sys.path.insert(0, str(learner_file.parent))
    module_spec = importlib.util.spec_from_file_location("submitted_learner", learner_file)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    learner = module.make_learner(dict(spec))
    n = int(spec["n_hidden"])

    def reply():
        w, b = learner.readout()
        w = np.asarray(w, dtype="<f8")
        b = np.asarray(b, dtype="<f8")
        if w.shape != (n, 2) or b.shape != (2,):
            w, b = np.full((n, 2), np.nan), np.full(2, np.nan)
        write_all(out_fd, np.ascontiguousarray(w).tobytes() + np.ascontiguousarray(b).tobytes())

    reply()
    while True:
        (steps,) = struct.unpack("<i", read_exact(0, 4))
        if steps <= 0:
            break
        trace = np.frombuffer(read_exact(0, steps * n * 4), dtype="<f4").reshape(steps, n).copy()
        p_forward = np.frombuffer(read_exact(0, steps * 8), dtype="<f8").copy()
        action = np.frombuffer(read_exact(0, steps), dtype="i1").astype(np.int64)
        reward = np.frombuffer(read_exact(0, steps * 8), dtype="<f8").copy()
        learner.learn({"trace": trace, "p_forward": p_forward, "action": action, "reward": reward})
        reply()


if __name__ == "__main__":
    main()
