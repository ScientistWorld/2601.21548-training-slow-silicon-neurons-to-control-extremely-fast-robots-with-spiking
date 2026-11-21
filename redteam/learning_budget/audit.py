#!/usr/bin/env python3
"""Final hardening audit for the learning-budget gym (one job, resumable).

Stages (each skipped when its result file exists):
  smoke      env regression, chip timing, protocol and failure handling at tiny scale
  select     Eq. 1 learning-rate sweep on the public split -> method/selected_hyperparameters.json
  canonical  reference + fixed baselines alone: scripts/evaluate.sh, scripts/evaluate_test.sh
  audit      alternative-learner sweep and public-readout trainer (visible split only),
             then reference rerun, better/weaker variants and shortcuts through the hidden evaluator
  analysis   paired per-chip comparisons and cohort ranks
No hidden chip, launch or outcome is used by any selection step.
"""
from __future__ import annotations
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent import futures
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from method.install import install  # noqa: E402
from method import select_learning_rate as selection  # noqa: E402

WORK = ROOT / "tmp" / "learning_budget"
RESULTS = HERE / "results"
LEARNERS = HERE / "learners"
PY = sys.executable
EXPERIMENTS = ("randomized_air_hockey", "velocity_range_wide", "velocity_range_extreme")


def available_cores():
    cores = len(os.sched_getaffinity(0))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cores = min(cores, max(1, int(float(quota) / float(period))))
    except (OSError, ValueError):
        pass
    return max(1, cores)


CORES = available_cores()
ENV = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def run(cmd):
    log("run", " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True, env=ENV, cwd=ROOT)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


OWNER = WORK / "owner.json"


def claim_workspace():
    """Worker drains can leave an older copy of this job running on another
    node while the scheduler restarts it. The newest instance owns the
    workspace; an older instance notices within seconds and kills its own
    process group (its evaluators and learner processes). All results are
    deterministic, so a superseded instance can only waste CPU, never change
    a number; this only removes the contention and file races."""
    os.setpgrp()
    token = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    write_json(OWNER, {"token": token, "claimed": time.time()})
    time.sleep(20)

    def current():
        try:
            return read_json(OWNER)["token"]
        except Exception:
            return token

    if current() != token:
        log("superseded before starting")
        raise SystemExit(0)

    def watch():
        while True:
            time.sleep(15)
            if current() != token:
                log("superseded by", current(), "- stopping this instance")
                os.killpg(os.getpgrp(), signal.SIGKILL)
    threading.Thread(target=watch, daemon=True).start()
    log("owner", token)


def read_json(path):
    return json.loads(Path(path).read_text())


def evaluate(split, outputs, scores, *extra):
    if Path(scores).exists():
        return read_json(scores)
    run([PY, ROOT / "eval" / split / "evaluate.py", "--outputs", outputs, "--scores", scores, *extra])
    return read_json(scores)


def selected():
    return read_json(ROOT / "method" / "selected_hyperparameters.json")["default"]


def learner_dir(root, name, source, config=None, extra_files=()):
    target = Path(root) / name
    shutil.rmtree(target, ignore_errors=True)
    if source is None:
        target.mkdir(parents=True)
        return target
    install(name, config or {}, root, source)
    for src, dst in extra_files:
        if Path(src).is_dir():
            shutil.copytree(src, target / dst)
        else:
            shutil.copyfile(src, target / dst)
    return target


# --------------------------------------------------------------------------- smoke
def stage_smoke():
    out = RESULTS / "smoke.json"
    if out.exists():
        return
    root = WORK / "smoke"
    if root.exists():
        shutil.rmtree(root)
    outputs = root / "outputs"
    ref = {"default": {"learning_rate": 0.05}}
    learner_dir(outputs, "reference", ROOT / "method" / "eprop_learner.py", ref)
    for name in ("nan_readout", "crash_wide", "bad_shape", "huge_readout", "forged_files", "bias_only"):
        learner_dir(outputs, name, LEARNERS / f"{name}.py")
    learner_dir(outputs, "missing_learner", None)
    record = {"cores": CORES, "affinity": len(os.sched_getaffinity(0)), "cpu_count": os.cpu_count(),
              "python": sys.version}
    try:
        record["cgroup_cpu_max"] = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    except OSError:
        pass
    probe = subprocess.run([PY, HERE / "probe.py", outputs / "reference"], check=True, env=ENV, cwd=ROOT,
                           capture_output=True, text=True)
    record["probe"] = json.loads(probe.stdout.strip().splitlines()[-1])
    log("probe", json.dumps(record["probe"]))
    workers = str(min(CORES, 6))
    for split in ("train", "test"):
        scores = evaluate(split, outputs, root / f"scores_{split}.json", "--chips", "2", "--budget", "12",
                          "--workers", workers)
        status = {m: d["status"] for m, d in scores["experiments"]["velocity_range_extreme"]["diagnostics"].items()}
        record[f"status_{split}"] = status
        expected_ok = {"reference", "forged_files", "bias_only", "random_policy", "always_forward"}
        for method, state in status.items():
            if (state == "ok") != (method in expected_ok):
                raise SystemExit(f"smoke {split}: unexpected status {method}={state}")
        for e in EXPERIMENTS:
            for method, row in scores["experiments"][e]["results"].items():
                if method not in expected_ok and any(v != 0 for v in row.values()):
                    raise SystemExit(f"smoke {split}: failed method {method} not at worst values")
    regression = record["probe"]["env_regression"]
    if abs(regression["forward"] - 0.199) > 0.05 or abs(regression["random"] - 0.076) > 0.05:
        raise SystemExit(f"slot environment departs from the previous physics: {regression}")
    write_json(out, record)


# --------------------------------------------------------------------------- selection
def stage_select():
    out = RESULTS / "selection.json"
    if out.exists():
        return
    parallel = max(1, CORES // 3)
    grid = list(selection.GRID)
    sweep_dir = WORK / "lr_sweep"
    for _ in range(3):
        best, table = selection.sweep("eq1", grid, {}, sweep_dir, parallel)
        rate = table[best]["config"]["learning_rate"]
        if rate == max(grid):
            grid.append(2 * rate)
        elif rate == min(grid):
            grid.insert(0, rate / 2)
        else:
            break
    record = {"default": table[best]["config"], "selected": best,
              "criterion": "weighted mean over experiments of (online_success + final_success) / 2 on the public split",
              "candidates": table}
    write_json(ROOT / "method" / "selected_hyperparameters.json", record)
    write_json(out, record)
    log("selected", best, json.dumps({n: round(r["objective"], 4) for n, r in table.items()}))


# --------------------------------------------------------------------------- canonical
def stage_canonical():
    """Reference + fixed baselines through the normal no-argument entry points."""
    outputs = ROOT / "outputs"
    if not (RESULTS / "canonical_scores_train.json").exists() or not (RESULTS / "canonical_scores_test.json").exists():
        shutil.rmtree(outputs, ignore_errors=True)
        run(["bash", ROOT / "scripts" / "method.sh"])
    if not (RESULTS / "canonical_scores_train.json").exists():
        run(["bash", ROOT / "scripts" / "evaluate.sh"])
        write_json(RESULTS / "canonical_scores_train.json", read_json(ROOT / "scoring" / "scores_train.json"))
    if not (RESULTS / "canonical_scores_test.json").exists():
        run(["bash", ROOT / "scripts" / "evaluate_test.sh"])
        test = read_json(ROOT / "scoring" / "scores_test.json")
        reference = read_json(ROOT / "scoring" / "reference.json")
        for e, spec in reference["experiments"].items():
            spec["results"] = {m: {"type": "proposed" if m == "spiking_eprop" else "baseline", **row}
                               for m, row in test["experiments"][e]["results"].items()}
        write_json(ROOT / "scoring" / "reference.json", reference)
        write_json(RESULTS / "canonical_scores_test.json", test)


# --------------------------------------------------------------------------- variants
def variant_plan(critic_config):
    best = selected()
    readouts = WORK / "public_trainer" / "outputs" / "public_trainer" / "public_readouts"
    ref_src = ROOT / "method" / "eprop_learner.py"
    plan = {
        # name: (kind, source, config, extra files)
        "spiking_eprop": ("reference rerun", ref_src, {"default": best}, ()),
        "eprop_tail_greedy": ("simple trick: freeze and sharpen readout (x4) for the last 500 episodes", ref_src,
                              {"default": {**best, "tail_episodes": 500, "tail_gain": 4.0}}, ()),
        "eprop_slow": ("weaker: Eq. 1 learning rate / 4", ref_src,
                       {"default": {**best, "learning_rate": best["learning_rate"] / 4}}, ()),
        "no_learning": ("weaker: readout never changes", LEARNERS / "no_learning.py", {}, ()),
        "bias_only": ("shortcut: state-blind constant-probability learner", LEARNERS / "bias_only.py", {}, ()),
        "forged_files": ("shortcut: self-reported perfect metrics", LEARNERS / "forged_files.py", {}, ()),
        "crash_wide": ("shortcut: skip one experiment by crashing", LEARNERS / "crash_wide.py", {}, ()),
        "nan_readout": ("shortcut: impossible readout values", LEARNERS / "nan_readout.py", {}, ()),
        "huge_readout": ("shortcut: out-of-range readout", LEARNERS / "huge_readout.py", {}, ()),
        "bad_shape": ("shortcut: readout larger than the chip", LEARNERS / "bad_shape.py", {}, ()),
        "missing_learner": ("shortcut: empty submission", None, {}, ()),
    }
    if critic_config is not None:
        plan["eprop_critic"] = ("better: value-baseline credit assignment (selected on public split)", ref_src,
                                {"default": critic_config}, ())
        plan["public_copy"] = ("shortcut: reuse readouts learned in 2000 trials on public chips, frozen",
                               LEARNERS / "public_copy.py", {}, ((readouts, "public_readouts"),))
    return plan


def stage_audit():
    """Public-split alternatives and the hidden-test audit, scheduled together.

    Visible-split work (alternative-learner sweep, public-readout trainer) and
    independent hidden-test variants share one pool of evaluator slots; the
    two variants that depend on public results start once those finish.
    """
    out = RESULTS / "audit_scores_test.json"
    if out.exists():
        return
    slots = max(1, (CORES + 1) // 3)
    best = selected()
    critic_grid = sorted({round(best["learning_rate"] * f, 6) for f in (1, 2, 4, 8)})
    root = WORK / "audit"
    trainer_root = WORK / "public_trainer"
    copy_root = WORK / "public_copy_train"

    def hidden(name, plan):
        kind, source, config, extra = plan[name]
        scores = root / name / "scores_test.json"
        if not scores.exists():
            learner_dir(root / name / "outputs", name, source, config, extra)
        return name, evaluate("test", root / name / "outputs", scores, "--no-baselines", "--workers", "3")

    def trainer():
        scores = trainer_root / "scores_train.json"
        if not scores.exists():
            learner_dir(trainer_root / "outputs", "public_trainer", LEARNERS / "public_trainer.py",
                        {"default": best}, [(ROOT / "method" / "eprop_learner.py", "eprop_base.py")])
        return evaluate("train", trainer_root / "outputs", scores, "--no-baselines", "--workers", "3")

    def critic(rate):
        config = {"default": {"learning_rate": rate, "gamma": selection.GAMMA, "critic_rate": 1.0}}
        return selection.run_candidate(WORK / "critic_sweep", selection.candidate_name("critic", rate), config)

    first = variant_plan(None)
    expensive = ["spiking_eprop", "eprop_tail_greedy", "eprop_slow", "bias_only", "forged_files", "no_learning",
                 "crash_wide"]
    cheap = [n for n in first if n not in expensive]
    done = {}
    with futures.ThreadPoolExecutor(max_workers=slots) as pool:
        critic_futures = [pool.submit(critic, r) for r in critic_grid]
        trainer_future = pool.submit(trainer)
        hidden_futures = [pool.submit(hidden, n, first) for n in expensive + cheap]
        critic_scores = dict(f.result() for f in critic_futures)
        objectives = {n: selection.objective(sc, n) for n, sc in critic_scores.items()}
        critic_best = max(objectives, key=objectives.get)
        critic_config = {"learning_rate": float(critic_best.split("_lr")[1]), "gamma": selection.GAMMA,
                         "critic_rate": 1.0}
        trainer_future.result()
        plan = variant_plan(critic_config)
        late = [pool.submit(hidden, n, plan) for n in ("eprop_critic", "public_copy")]
        copy_scores = copy_root / "scores_train.json"
        if not copy_scores.exists():
            learner_dir(copy_root / "outputs", "public_copy", LEARNERS / "public_copy.py", {},
                        [(trainer_root / "outputs" / "public_trainer" / "public_readouts", "public_readouts")])
        copy_future = pool.submit(evaluate, "train", copy_root / "outputs", copy_scores, "--no-baselines",
                                  "--workers", "3")
        for f in hidden_futures + late:
            name, scores = f.result()
            done[name] = scores
        copy_train = copy_future.result()
    write_json(RESULTS / "public_split.json", {
        "critic_selected": critic_best,
        "critic_candidates": {n: {"objective": objectives[n], "results": {e: critic_scores[n]["experiments"][e]["results"][n]
                                                                          for e in EXPERIMENTS}} for n in critic_scores},
        "public_trainer_train": read_json(trainer_root / "scores_train.json"),
        "public_copy_train": copy_train})
    merged = {"experiments": {e: {"results": {}, "diagnostics": {}} for e in EXPERIMENTS}, "kinds": {}}
    for name, scores in done.items():
        merged["kinds"][name] = plan[name][0]
        for e in EXPERIMENTS:
            merged["experiments"][e]["results"][name] = scores["experiments"][e]["results"][name]
            merged["experiments"][e]["diagnostics"][name] = scores["experiments"][e]["diagnostics"][name]
    write_json(out, merged)


# --------------------------------------------------------------------------- analysis
def per_chip(diag):
    return diag.get("per_chip_online_success"), diag.get("per_chip_final_success")


def stage_analysis():
    import numpy as np
    sys.path.insert(0, str(ROOT / "eval"))
    from rank import rank
    reference = read_json(ROOT / "scoring" / "reference.json")
    canonical = read_json(RESULTS / "canonical_scores_test.json")
    audit = read_json(RESULTS / "audit_scores_test.json")
    weights = {e: reference["experiments"][e]["weight"] for e in EXPERIMENTS}
    cohort = {"experiments": {}}
    for e in EXPERIMENTS:
        results = dict(canonical["experiments"][e]["results"])
        results.update({f"audit:{m}": r for m, r in audit["experiments"][e]["results"].items()})
        cohort["experiments"][e] = {"results": results}
    ranks = rank(cohort, reference)
    # Paired per-chip comparison against the canonical reference (same chips,
    # launches and action draws). Combined gain uses the rank's experiment
    # weights with raw (unnormalised) success differences.
    ref_diag = {e: canonical["experiments"][e]["diagnostics"]["spiking_eprop"] for e in EXPERIMENTS}
    rng = np.random.default_rng(0)
    paired = {}
    for name in audit["kinds"]:
        diags = {e: audit["experiments"][e]["diagnostics"][name] for e in EXPERIMENTS}
        if any(d.get("status") != "ok" for d in diags.values()):
            paired[name] = {"status": "failed", "reason": next(d.get("reason") for d in diags.values()
                                                                 if d.get("status") != "ok")}
            continue
        row = {"status": "ok"}
        combined = 0.0
        for metric, key in (("online_success", "per_chip_online_success"), ("final_success", "per_chip_final_success")):
            per = np.zeros(len(ref_diag[EXPERIMENTS[0]][key]))
            for e in EXPERIMENTS:
                diff = np.array(diags[e][key]) - np.array(ref_diag[e][key])
                row[f"{e}:{metric}:mean_diff"] = float(diff.mean())
                row[f"{e}:{metric}:chip_sd"] = float(diff.std(ddof=1))
                per += weights[e] * diff
            combined = combined + per
            boot = per[rng.integers(0, len(per), (10000, len(per)))].mean(axis=1)
            row[f"weighted:{metric}"] = {"mean": float(per.mean()), "ci95": [float(np.quantile(boot, .025)),
                                         float(np.quantile(boot, .975))], "chips_positive": int((per > 0).sum()),
                                         "chips_negative": int((per < 0).sum())}
        boot = combined[rng.integers(0, len(combined), (10000, len(combined)))].mean(axis=1)
        row["weighted:both"] = {"mean": float(combined.mean()), "ci95": [float(np.quantile(boot, .025)),
                                float(np.quantile(boot, .975))], "chips_positive": int((combined > 0).sum())}
        paired[name] = row
    # Reference determinism and train/test comparability.
    rerun = {e: audit["experiments"][e]["results"].get("spiking_eprop") for e in EXPERIMENTS}
    canon = {e: canonical["experiments"][e]["results"]["spiking_eprop"] for e in EXPERIMENTS}
    train = read_json(RESULTS / "canonical_scores_train.json")
    chip_noise = {}
    for e in EXPERIMENTS:
        for key in ("per_chip_online_success", "per_chip_final_success"):
            values = np.array(ref_diag[e][key])
            chip_noise[f"{e}:{key}"] = {"chip_sd": float(values.std(ddof=1)),
                                        "standard_error": float(values.std(ddof=1) / math.sqrt(len(values)))}
    summary = {
        "ranks": ranks,
        "paired_vs_reference": paired,
        "reference_rerun_identical": rerun == canon,
        "reference_test": canon,
        "reference_train": {e: train["experiments"][e]["results"]["spiking_eprop"] for e in EXPERIMENTS},
        "baselines_test": {e: {m: r for m, r in canonical["experiments"][e]["results"].items() if m != "spiking_eprop"}
                           for e in EXPERIMENTS},
        "reference_chip_noise": chip_noise,
        "selection": read_json(RESULTS / "selection.json")["candidates"],
        "public_split": {k: v for k, v in read_json(RESULTS / "public_split.json").items()
                         if k in ("critic_selected", "critic_candidates")},
        "public_trainer_train": {e: read_json(RESULTS / "public_split.json")["public_trainer_train"]["experiments"][e]["results"]
                                 for e in EXPERIMENTS},
        "public_copy_train": {e: read_json(RESULTS / "public_split.json")["public_copy_train"]["experiments"][e]["results"]
                              for e in EXPERIMENTS},
        "variant_kinds": audit["kinds"],
    }
    write_json(RESULTS / "summary.json", summary)
    log("ranks", json.dumps({k: round(v, 4) for k, v in ranks.items()}))


def main():
    stages = sys.argv[1:] or ["smoke", "select", "canonical", "audit", "analysis"]
    claim_workspace()
    log("cores", CORES, "stages", stages)
    for stage in stages:
        log("stage", stage)
        globals()[f"stage_{stage}"]()
    log("audit complete")


if __name__ == "__main__":
    main()
