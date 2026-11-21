"""Held-out test scoring: online learning on hidden controller instances.

Each method directory ``outputs/<method>/`` must contain ``learner.py`` with
``make_learner(spec)``. For every experiment the evaluator builds hidden
chip instances, starts one learner process per chip (eval/test/learner_host.py)
and runs ``BUDGET`` sequential episodes on each chip. The evaluator owns the
robot, the chips, the episodes, the action sampling and every outcome: before
each episode the learner supplies a linear readout ``(w, b)``; the evaluator
computes the softmax forward probability from the chip's filtered spike trains,
samples the action from common random numbers, and after the episode sends the
learner its own record (spike trains, probabilities, actions, Eq. 2 rewards).

Scores (higher is better):
  online_success  fraction of all BUDGET online episodes that succeed;
  final_success   fraction of the last quarter of online episodes that succeed.
A malformed readout, a crash, a timeout or a missing learner fails the whole
method: it receives 0 in every scored metric of every experiment.

The hidden chips, launches and action draws are generated in memory from the
seeds below; learners never receive chip arrays, launches or robot states.
The proposed methods named in scoring/reference.json are always scored, so a
missing reference directory is a failure, not an omission. Independent of
eval/train/. Never imports method/.
"""
from __future__ import annotations
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
import argparse
import json
import multiprocessing as mp
import select
import struct
import subprocess
import sys
import tempfile
import time
from concurrent import futures
from pathlib import Path
import numpy as np
from eval.test.task_sim import AirHockeySlots, sample_initial_conditions
from eval.test.chip_runtime import ChipBank, build_chip

ROOT = Path(__file__).resolve().parents[2]
HOST = Path(__file__).resolve().with_name("learner_host.py")
OUTPUTS = ROOT / "outputs"
SCORES = ROOT / "scoring" / "scores_test.json"
RAW = Path(__file__).with_name("raw")
EXPERIMENTS = {
    "randomized_air_hockey": "random_position_speed",
    "velocity_range_wide": "speed_wide",
    "velocity_range_extreme": "speed_extreme",
}
N_HIDDEN = 1020
N_CHIPS = 20
BUDGET = 2000
MAX_TICKS = 100
CHIP_SEED = 3_326_147_246_915_709_710
EPISODE_SEED = 184_885_199_261_487_550
BASELINES = {"random_policy": .5, "always_forward": 1.}
CALL_TIMEOUT_S = 600.
READOUT_BOUND = 1e6
METRICS = ("online_success", "final_success")


class LearnerFailure(RuntimeError):
    pass


def worst():
    return {m: 0. for m in METRICS}


def chips_for(experiment, n_chips):
    i = list(EXPERIMENTS).index(experiment)
    return [build_chip(EXPERIMENTS[experiment], [CHIP_SEED, i, k], N_HIDDEN) for k in range(n_chips)]


def episode_plan(experiment, n_chips, budget):
    i = list(EXPERIMENTS).index(experiment)
    inits = np.stack([sample_initial_conditions(EXPERIMENTS[experiment],
                      np.random.default_rng([EPISODE_SEED, i, k, 0]), budget) for k in range(n_chips)])
    # Common random numbers: every method sees the same uniform draw at a given
    # chip, episode and tick, so action sampling is paired across methods.
    draws = np.stack([np.random.default_rng([EPISODE_SEED, i, k, 1]).random((budget, MAX_TICKS), dtype=np.float32)
                      for k in range(n_chips)])
    return inits, draws


class LearnerClient:
    def __init__(self, learner_file, spec):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        self.n = int(spec["n_hidden"])
        self.proc = subprocess.Popen([sys.executable, str(HOST), str(learner_file), json.dumps(spec)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     cwd=str(learner_file.parent), env=env)
        self.rfd, self.wfd = self.proc.stdout.fileno(), self.proc.stdin.fileno()
        os.set_blocking(self.wfd, False)

    def _read(self, nbytes):
        deadline, parts, got = time.monotonic() + CALL_TIMEOUT_S, [], 0
        while got < nbytes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LearnerFailure("learner did not answer within the call limit")
            ready, _, _ = select.select([self.rfd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(self.rfd, nbytes - got)
            if not chunk:
                raise LearnerFailure(f"learner process exited (code {self.proc.poll()})")
            parts.append(chunk)
            got += len(chunk)
        return b"".join(parts)

    def _write(self, data, timeout=CALL_TIMEOUT_S):
        deadline, view = time.monotonic() + timeout, memoryview(data)
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LearnerFailure("learner did not read its input within the call limit")
            _, ready, _ = select.select([], [self.wfd], [], remaining)
            if not ready:
                continue
            try:
                view = view[os.write(self.wfd, view):]
            except BlockingIOError:
                continue
            except OSError as exc:
                raise LearnerFailure(f"learner input closed: {exc}") from exc

    def send(self, trace, p_forward, action, reward):
        self._write(struct.pack("<i", len(action)) + np.ascontiguousarray(trace, dtype="<f4").tobytes()
                    + np.ascontiguousarray(p_forward, dtype="<f8").tobytes()
                    + np.ascontiguousarray(action, dtype="i1").tobytes()
                    + np.ascontiguousarray(reward, dtype="<f8").tobytes())

    def receive(self):
        values = np.frombuffer(self._read((2 * self.n + 2) * 8), dtype="<f8")
        if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > READOUT_BOUND:
            raise LearnerFailure("readout is not finite or exceeds the bound")
        return values[:2 * self.n].reshape(self.n, 2).copy(), values[2 * self.n:].copy()

    def close(self):
        try:
            if self.proc.poll() is None:
                self._write(struct.pack("<i", 0), timeout=5.)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def run_unit(method, experiment, method_dir, n_chips, budget):
    """Run ``budget`` online episodes on every chip; return raw outcomes."""
    inits, draws = episode_plan(experiment, n_chips, budget)
    C = n_chips
    env = AirHockeySlots(C)
    baseline = method_dir is None
    learners = []
    wdiff = np.zeros((C, N_HIDDEN))
    bdiff = np.zeros(C)
    try:
        if baseline:
            p_const = BASELINES[method]
        else:
            learner_file = (Path(method_dir) / "learner.py").resolve()
            if not learner_file.is_file():
                raise LearnerFailure("missing learner.py")
            bank = ChipBank(chips_for(experiment, C))
            for k in range(C):
                learners.append(LearnerClient(learner_file, {
                    "experiment": experiment, "n_hidden": N_HIDDEN, "n_actions": 2, "budget": budget,
                    "chip": k, "seed": 1000 * list(EXPERIMENTS).index(experiment) + k,
                    "workdir": str(Path(method_dir).resolve())}))
            for k in range(C):
                w, b = learners[k].receive()
                wdiff[k], bdiff[k] = w[:, 1] - w[:, 0], b[1] - b[0]
            bank.reset(np.arange(C))
            tbuf = np.zeros((C, MAX_TICKS, N_HIDDEN), dtype=np.float32)
        success = np.zeros((C, budget), dtype=bool)
        hit = np.zeros((C, budget), dtype=bool)
        steps = np.zeros((C, budget), dtype=np.int16)
        pbuf = np.zeros((C, MAX_TICKS))
        abuf = np.zeros((C, MAX_TICKS), dtype=np.int8)
        rbuf = np.zeros((C, MAX_TICKS))
        ep = np.zeros(C, dtype=int)
        tick = np.zeros(C, dtype=int)
        active = np.ones(C, dtype=bool)
        env.reset(np.arange(C), inits[:, 0])
        while active.any():
            rows = np.flatnonzero(active)
            if baseline:
                p = np.full(C, p_const)
            else:
                trace = bank.step(env.obs)
                logit = np.einsum("cn,cn->c", trace.astype(np.float64), wdiff) + bdiff
                p = .5 * (1. + np.tanh(.5 * logit))
                tbuf[rows, tick[rows]] = trace[rows]
            a = (draws[rows, ep[rows], tick[rows]] < p[rows]).astype(np.int64)
            pbuf[rows, tick[rows]] = p[rows]
            abuf[rows, tick[rows]] = a
            reward, done = env.step(rows, a)
            rbuf[rows, tick[rows]] = reward
            tick[rows] += 1
            ended = rows[done]
            if not len(ended):
                continue
            success[ended, ep[ended]] = env.success[ended]
            hit[ended, ep[ended]] = env.hit[ended]
            steps[ended, ep[ended]] = tick[ended]
            if not baseline:
                for c in ended:
                    T = tick[c]
                    learners[c].send(tbuf[c, :T], pbuf[c, :T], abuf[c, :T], rbuf[c, :T])
                for c in ended:
                    w, b = learners[c].receive()
                    wdiff[c], bdiff[c] = w[:, 1] - w[:, 0], b[1] - b[0]
            ep[ended] += 1
            active[ended[ep[ended] >= budget]] = False
            cont = ended[ep[ended] < budget]
            if len(cont):
                env.reset(cont, inits[cont, ep[cont]])
                if not baseline:
                    bank.reset(cont)
                tick[cont] = 0
    finally:
        for learner in learners:
            learner.close()
    return {"success": success, "hit": hit, "steps": steps}


def run_unit_safe(task):
    method, experiment, method_dir, n_chips, budget = task
    start = time.time()
    try:
        outcome = run_unit(method, experiment, method_dir, n_chips, budget)
        return method, experiment, "ok", outcome, time.time() - start
    except Exception as exc:  # any failure fails the method
        return method, experiment, f"failed: {type(exc).__name__}: {exc}", None, time.time() - start


def summarize(outcome, budget):
    s = np.asarray(outcome["success"], dtype=float)
    window = max(1, budget // 4)
    return {"online_success": float(s.mean()), "final_success": float(s[:, -window:].mean())}


def diagnostics(outcome, budget):
    s = np.asarray(outcome["success"], dtype=float)
    window = max(1, budget // 4)
    return {"status": "ok",
            "per_chip_online_success": s.mean(axis=1).round(6).tolist(),
            "per_chip_final_success": s[:, -window:].mean(axis=1).round(6).tolist(),
            "hit_rate": float(np.asarray(outcome["hit"]).mean()),
            "mean_episode_ticks": float(np.asarray(outcome["steps"]).mean())}


def available_cores():
    cores = len(os.sched_getaffinity(0))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cores = min(cores, max(1, int(float(quota) / float(period))))
    except (OSError, ValueError):
        pass
    return max(1, cores)


def discover_methods(outputs):
    names = {p.name for p in outputs.iterdir() if p.is_dir()} if outputs.exists() else set()
    if outputs.resolve() == OUTPUTS.resolve():
        reference = json.loads((ROOT / "scoring" / "reference.json").read_text())
        for spec in reference["experiments"].values():
            names.update(m for m, r in spec["results"].items() if r.get("type") == "proposed")
    return sorted(names - set(BASELINES))


def save_raw(method, experiment, outcome, scores_path):
    # Paired per-episode outcomes stay inside this split's private area.
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / f"{scores_path.stem}__{method}__{experiment}.npz"
    fd, temporary = tempfile.mkstemp(dir=RAW, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **outcome)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--outputs", type=Path, default=OUTPUTS, help="directory of method directories")
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--methods", nargs="*", help="development only: score only these method directories")
    parser.add_argument("--no-baselines", action="store_true", help="development only")
    parser.add_argument("--chips", type=int, default=N_CHIPS, help="development only")
    parser.add_argument("--budget", type=int, default=BUDGET, help="development only")
    parser.add_argument("--workers", type=int, default=0, help="parallel (method, experiment) units")
    args = parser.parse_args(argv)
    methods = args.methods if args.methods else discover_methods(args.outputs)
    baselines = [] if args.no_baselines else list(BASELINES)
    tasks = [(m, e, str(args.outputs / m), args.chips, args.budget) for m in methods for e in EXPERIMENTS]
    tasks += [(m, e, None, args.chips, args.budget) for m in baselines for e in EXPERIMENTS]
    workers = args.workers or min(len(tasks), available_cores(), 4)
    results = {}
    with futures.ProcessPoolExecutor(max_workers=max(1, workers), mp_context=mp.get_context("spawn")) as pool:
        for method, experiment, status, outcome, seconds in pool.map(run_unit_safe, tasks):
            results[method, experiment] = (status, outcome)
            print(f"{method} {experiment} {status} ({seconds:.0f}s)", flush=True)
    scores = {e: {"results": {}, "diagnostics": {}} for e in EXPERIMENTS}
    for method in methods + baselines:
        rows = [results[method, e] for e in EXPERIMENTS]
        reason = next((status for status, _ in rows if status != "ok"), None)
        for e, (status, outcome) in zip(EXPERIMENTS, rows):
            if reason is None:
                save_raw(method, e, outcome, args.scores)
                scores[e]["results"][method] = summarize(outcome, args.budget)
                scores[e]["diagnostics"][method] = diagnostics(outcome, args.budget)
            else:
                # Any failed experiment fails the whole method in every experiment.
                scores[e]["results"][method] = worst()
                scores[e]["diagnostics"][method] = {"status": "failed", "reason": status if status != "ok" else reason}
    atomic_write(args.scores, json.dumps({"slice": "test", "experiments": scores}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
