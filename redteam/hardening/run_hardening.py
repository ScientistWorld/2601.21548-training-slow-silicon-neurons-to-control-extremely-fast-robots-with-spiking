#!/usr/bin/env python3
"""Hardening job: reproduce the reference, run variants and shortcuts through the
unmodified evaluators, and regenerate the final reference-only scores.

Every stage writes ``redteam/hardening/results/<stage>.json`` and is skipped
when that file exists, so a killed job resumes where it stopped.  Candidate
and variant controllers live in ``redteam/hardening/_staging`` (gitignored) and
are linked into ``outputs/`` only while being scored.  ``outputs/spiking_eprop``
is hidden as ``outputs/_ref_hold`` (names starting with ``_`` are not scored)
during shortcut cases and always restored.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path("/home/user")
sys.path.insert(0, str(ROOT))
HERE = ROOT / "redteam" / "hardening"
STAGING = HERE / "_staging"
RESULTS = HERE / "results"
OUTPUTS = ROOT / "outputs"
REF = OUTPUTS / "spiking_eprop"
HOLD = OUTPUTS / "_ref_hold"
EXPERIMENTS = ["randomized_air_hockey", "velocity_range_wide", "velocity_range_extreme", "single_core_extreme"]
SEEDS = [0, 1, 2, 3, 4]
LR_CANDIDATES = [0.007, 0.012, 0.025]
CRITIC_LR = 0.1
ENV = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 60:7.1f} min] {msg}", flush=True)


def workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def done(stage: str) -> bool:
    return (RESULTS / f"{stage}.json").exists()


def load(stage: str):
    return json.loads((RESULTS / f"{stage}.json").read_text())


def record(stage: str, payload) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")


def weights() -> dict[str, float]:
    ref = json.loads((ROOT / "scoring" / "reference.json").read_text())
    return {e: float(ref["experiments"][e]["weight"]) for e in EXPERIMENTS}


def run(cmd: list[str], check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True)
    if not quiet or res.returncode:
        if res.stdout.strip():
            print(res.stdout.rstrip()[-4000:], flush=True)
        if res.stderr.strip():
            print(res.stderr.rstrip()[-4000:], flush=True)
    if check and res.returncode:
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(cmd)}")
    return res


def run_pool(cmds: list[list[str]]) -> list[list[str]]:
    """Run commands with as many concurrent processes as CPUs; return failures."""
    n = workers()
    pending = list(cmds)
    active: list[tuple[subprocess.Popen, list[str]]] = []
    failures = []
    while pending or active:
        while pending and len(active) < n:
            cmd = pending.pop(0)
            active.append((subprocess.Popen(cmd, cwd=ROOT, env=ENV), cmd))
        time.sleep(1.0)
        for proc, cmd in list(active):
            if proc.poll() is not None:
                active.remove((proc, cmd))
                if proc.returncode:
                    failures.append(cmd)
    if failures:
        log(f"{len(failures)} training commands failed: {failures[:3]}")
    return failures


def pmap(fn, items: list) -> list:
    n = min(workers(), max(1, len(items)))
    if n == 1:
        return [fn(x) for x in items]
    import multiprocessing
    with ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("fork")) as pool:
        return list(pool.map(fn, items))


def train_cmds(out_dir: Path, lr: float, episodes: int, extra: list[str] | None = None,
               snapshots: list[str] | None = None, seeds=SEEDS, experiments=EXPERIMENTS) -> list[list[str]]:
    cmds = []
    for exp in experiments:
        for seed in seeds:
            cmd = [sys.executable, "method/train.py", "--experiments", exp, "--seeds", str(seed),
                   "--episodes", str(episodes), "--learning-rate", repr(lr), "--output-dir", str(out_dir)]
            for snap in snapshots or []:
                cmd += ["--snapshot", snap]
            cmds.append(cmd + (extra or []))
    return cmds


def ref_dir(lr: float, episodes: int) -> Path:
    return STAGING / f"ref_lr{lr}" / f"ep{episodes}"


def restore_reference() -> None:
    if HOLD.exists() and not REF.exists():
        HOLD.rename(REF)


def uninstall_case() -> None:
    if OUTPUTS.exists():
        for path in list(OUTPUTS.iterdir()):
            if path.is_symlink():
                path.unlink()
    restore_reference()


def evaluate(kinds=("test",)) -> dict:
    out = {}
    for kind in kinds:
        start = time.time()
        res = run(["bash", f"scripts/evaluate_{kind}.sh"], check=False, quiet=True)
        out[kind] = {"exit_code": res.returncode, "seconds": round(time.time() - start, 1)}
        path = ROOT / "scoring" / f"scores_{kind}.json"
        out[kind]["scores"] = json.loads(path.read_text()) if path.exists() else None
        if res.returncode:
            out[kind]["stderr_tail"] = res.stderr[-3000:]
        log(f"evaluate_{kind}.sh exit={res.returncode} in {out[kind]['seconds']}s")
    return out


# --------------------------------------------------------------------------- stages

def stage_smoke() -> None:
    if done("smoke"):
        return
    import data.air_hockey_env as pub_env
    import data.neuromorphic_chip as pub_chip
    import eval.test.chip_runtime as frz_chip
    import eval.test.evaluate as te
    import eval.test.task_sim as frz_env
    from method.spiking_eprop_rl import EpropConfig, train_controller

    def body(path: Path) -> str:
        text = path.read_text()
        return text[text.index('"""', 3) + 3:]

    same_src = {
        "env": body(ROOT / "data/air_hockey_env.py") in (ROOT / "eval/test/task_sim.py").read_text(),
        "chip": body(ROOT / "data/neuromorphic_chip.py") in (ROOT / "eval/test/chip_runtime.py").read_text(),
    }
    rng = np.random.default_rng(0)
    init = pub_env.sample_initial_conditions("speed_extreme", rng, 64)
    acts = rng.integers(0, 2, size=(100, 64))
    a, b = pub_env.AirHockeyBatch(init), frz_env.AirHockeyBatch(init)
    for k in range(100):
        oa, ob = a.step(acts[k]), b.step(acts[k])
    assert np.array_equal(oa[0], ob[0]) and np.array_equal(a.success, b.success)
    cfg = EpropConfig(hidden_neurons=64, cells_per_dim=4, calibration_episodes=4, critic_learning_rate=CRITIC_LR)
    config, _ = train_controller("random_position_speed", 0, 6, cfg)
    obs = pub_env.AirHockeyBatch(pub_env.sample_initial_conditions("speed_wide", rng, 8)).observe()
    assert np.array_equal(pub_chip.ChipBatch(config, 8).step(obs), frz_chip.ChipBatch(config, 8).step(obs))
    tasks = [("x", e, True) for e in EXPERIMENTS[:2]]
    smoke_scores = [te.score_task(t)[2] for t in tasks]
    record("smoke", {"same_source": same_src, "workers": workers(), "numpy": np.__version__,
                     "baseline_scores": smoke_scores})
    log(f"smoke ok: same_source={same_src} workers={workers()}")
    if not all(same_src.values()):
        raise RuntimeError("frozen test copies differ from data/ sources")


def stage_candidates() -> None:
    """Reference training for every learning-rate candidate (Eq. 1, 4000 episodes, snapshots)."""
    if done("candidates"):
        return
    cmds = []
    for lr in LR_CANDIDATES:
        snaps = [f"0={ref_dir(lr, 0)}", f"500={ref_dir(lr, 500)}", f"2000={ref_dir(lr, 2000)}"]
        cmds += train_cmds(ref_dir(lr, 4000), lr, 4000, snapshots=snaps)
    failures = run_pool(cmds)
    record("candidates", {"learning_rates": LR_CANDIDATES, "failed_commands": failures})


def _train_task(task):
    import eval.train.evaluate as tr
    lr, exp = task
    try:
        return lr, exp, tr.summarize(tr.evaluate_method_experiment(ref_dir(lr, 2000), exp)), None
    except Exception as exc:  # missing/diverged seeds
        return lr, exp, tr.worst(exp), str(exc)


def stage_select() -> float:
    """Pick the learning rate on the visible slice only (public validation episodes)."""
    if done("select_lr"):
        return load("select_lr")["chosen_learning_rate"]
    w = weights()
    rows = pmap(_train_task, [(lr, e) for lr in LR_CANDIDATES for e in EXPERIMENTS])
    table: dict = {}
    for lr, exp, metrics, err in rows:
        table.setdefault(str(lr), {})[exp] = dict(metrics, **({"error": err} if err else {}))
    score = {lr: sum(w[e] * (table[str(lr)][e]["success_rate"] + 0.3 * table[str(lr)][e]["seed_worst_success"])
                     for e in EXPERIMENTS) for lr in LR_CANDIDATES}
    chosen = max(LR_CANDIDATES, key=lambda lr: score[lr])
    if REF.exists():
        shutil.rmtree(REF)
    install_reference(chosen, SEEDS)
    curves = {}
    for lr in LR_CANDIDATES:
        for exp in EXPERIMENTS:
            for seed in SEEDS:
                p = ref_dir(lr, 4000) / exp / f"training_seed_{seed}.json"
                if p.exists():
                    h = json.loads(p.read_text())
                    o = np.array([int(c) for c in h["online_success"]] or [0])
                    curves.setdefault(str(lr), {}).setdefault(exp, {})[str(seed)] = {
                        "online_per_250": [round(float(o[i:i + 250].mean()), 3) for i in range(0, len(o), 250)],
                        "status": h.get("status", "ok"), "wall_seconds": round(h.get("wall_seconds", 0.0))}
    record("select_lr", {"train_slice_metrics": table, "selection_score": {str(k): v for k, v in score.items()},
                         "rule": "max over learning rates of sum_e weight_e * (success + 0.3 * seed_worst), visible slice",
                         "chosen_learning_rate": chosen, "online_curves": curves})
    log(f"learning-rate selection: {score} -> {chosen}")
    return chosen


def install_reference(lr: float, seeds) -> None:
    """Copy the 2000-episode snapshots of the chosen learning rate into outputs/spiking_eprop."""
    for exp in EXPERIMENTS:
        (REF / exp).mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            shutil.copy2(ref_dir(lr, 2000) / exp / f"seed_{seed}.npz", REF / exp / f"seed_{seed}.npz")
            hist = json.loads((ref_dir(lr, 4000) / exp / f"training_seed_{seed}.json").read_text())
            online = np.array([int(c) for c in hist["online_success"][:2000]])
            hist.update({"episodes": 2000, "online_success": hist["online_success"][:2000],
                         "online_hit": hist["online_hit"][:2000], "episode_steps": hist["episode_steps"][:2000],
                         "online_success_last200": float(online[-200:].mean()), "method": "spiking_eprop",
                         "note": "first 2000 episodes of the uninterrupted 4000-episode training run"})
            hist.pop("wall_seconds", None)
            (REF / exp / f"training_seed_{seed}.json").write_text(json.dumps(hist) + "\n")


SEEDS10 = list(range(10))


def stage_seeds10(lr: float) -> None:
    """Paper procedure: 10 reservoir seeds per run.  Train seeds 5-9 and the conjunctive variant."""
    if done("seeds10_training"):
        return
    new = [5, 6, 7, 8, 9]
    snaps = [f"0={ref_dir(lr, 0)}", f"500={ref_dir(lr, 500)}", f"2000={ref_dir(lr, 2000)}"]
    cmds = train_cmds(ref_dir(lr, 4000), lr, 4000, snapshots=snaps, seeds=new)
    cmds += train_cmds(STAGING / "spiking_eprop_critic", lr, 2000, seeds=new,
                       extra=["--critic-learning-rate", repr(CRITIC_LR)])
    for exp in EXPERIMENTS:
        for seed in SEEDS10:
            cmds.append([sys.executable, "redteam/hardening/variants.py", "--experiment", exp, "--seed", str(seed),
                         "--episodes", "2000", "--learning-rate", repr(lr),
                         "--output-dir", str(STAGING / "spiking_eprop_conjunctive")])
    failures = run_pool(cmds)
    install_reference(lr, new)
    pmap(_distill_task, [(lr, e, s) for e in EXPERIMENTS for s in new])
    record("seeds10_training", {"learning_rate": lr, "failed_commands": failures, "variants": {
        "spiking_eprop_conjunctive": "same Eq. 1 readout learning and budget; hidden units are coincidence "
                                     "detectors over (lateral puck position, lateral puck velocity) cells "
                                     "instead of random sparse input synapses",
    }})


def stage_extras(lr: float) -> None:
    if done("extras"):
        return
    cmds = train_cmds(STAGING / "cherry_seeds", lr, 2000, seeds=[5, 6, 7, 8, 9])
    cmds += train_cmds(STAGING / "spiking_eprop_20cells", lr, 2000, extra=["--cells-per-dim", "20"])
    cmds += train_cmds(STAGING / "spiking_eprop_critic", lr, 2000, extra=["--critic-learning-rate", repr(CRITIC_LR)])
    failures = run_pool(cmds)
    record("extras", {"learning_rate": lr, "failed_commands": failures, "variants": {
        "spiking_eprop_20cells": "same rule, reservoir size and budget; 20 instead of 10 population cells per state variable",
        "spiking_eprop_critic": "same reservoir; return minus a learned value readout (actor-critic e-prop), not Eq. 1",
        "cherry_seeds": "reference recipe with reservoir seeds 5-9, used for the seed cherry-picking shortcut",
    }})


def _seed_task(task):
    import eval.train.evaluate as tr
    from data.neuromorphic_chip import load_config
    exp, seed, path = task
    config = load_config(path)
    s, _ = tr.rollout_controller(config, tr.validation_episodes(exp))
    return exp, seed, float(s.mean())


def stage_cherry_pick() -> None:
    """Shortcut: train 10 seeds, submit the 5 with the best visible-slice success."""
    if done("cherry_pick"):
        return
    tasks = [(e, s, str(REF / e / f"seed_{s}.npz")) for e in EXPERIMENTS for s in SEEDS]
    tasks += [(e, s, str(STAGING / "cherry_seeds" / e / f"seed_{s}.npz")) for e in EXPERIMENTS for s in range(5, 10)
              if (STAGING / "cherry_seeds" / e / f"seed_{s}.npz").exists()]
    rows = pmap(_seed_task, tasks)
    out_dir = STAGING / "seed_cherry_pick"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    chosen = {}
    for exp in EXPERIMENTS:
        ranked = sorted([(v, s, p) for (e, s, v), (_, _, p) in zip(rows, tasks) if e == exp], reverse=True)[:5]
        (out_dir / exp).mkdir(parents=True)
        for k, (_, s, p) in enumerate(ranked):
            shutil.copy2(p, out_dir / exp / f"seed_{k}.npz")
        chosen[exp] = [(s, v) for v, s, _ in ranked]
    record("cherry_pick", {"train_success_per_seed": [list(r) for r in rows], "chosen": chosen})


def _distill_task(task):
    from data.neuromorphic_chip import load_config, save_config
    from method.train import EXPERIMENTS as PLAN
    from redteam.hardening.variants import distill_readout
    lr, exp, seed = task
    out = STAGING / "oracle_distilled" / exp / f"seed_{seed}.npz"
    if not out.exists():
        base = load_config(ref_dir(lr, 0) / exp / f"seed_{seed}.npz")
        save_config(out, distill_readout(base, PLAN[exp][0], seed))
    return exp, seed


def stage_distill(lr: float) -> None:
    if done("distill"):
        return
    pmap(_distill_task, [(lr, e, s) for e in EXPERIMENTS for s in SEEDS])
    record("distill", {"note": "reference reservoirs (untrained snapshot), readout fit by ridge regression to the "
                               "actions of a per-episode delay-enumeration teacher (out of bounds per DIRECTION.md)"})


def stage_eval_reference() -> None:
    if done("eval_reference"):
        return
    uninstall_case()
    record("eval_reference", evaluate(("train", "test")))


def variant_dirs(lr: float) -> dict[str, Path]:
    return {
        "spiking_eprop_untrained": ref_dir(lr, 0),
        "spiking_eprop_500ep": ref_dir(lr, 500),
        "spiking_eprop_4000ep": ref_dir(lr, 4000),
        "spiking_eprop_critic": STAGING / "spiking_eprop_critic",
        "spiking_eprop_conjunctive": STAGING / "spiking_eprop_conjunctive",
        "oracle_distilled": STAGING / "oracle_distilled",
    }


def stage_eval_variants(lr: float) -> None:
    if done("eval_variants"):
        return
    uninstall_case()
    for name, path in variant_dirs(lr).items():
        (OUTPUTS / name).symlink_to(path.resolve(), target_is_directory=True)
    try:
        res = evaluate(("test",))
    finally:
        uninstall_case()
    record("eval_variants", res)


def build_shortcut_cases() -> dict[str, Path]:
    from data.neuromorphic_chip import load_config, save_config
    from redteam.hardening.variants import constant_controller
    root = STAGING / "shortcuts"
    if root.exists():
        shutil.rmtree(root)
    cases = {}
    perfect = {"success_rate": 1.0, "hit_rate": 1.0, "controller_units": 1, "seed_worst_success": 1.0,
               "eval_success_by_seed": [1.0, 1.0, 1.0]}
    impossible = {"success_rate": 2.0, "hit_rate": 1e6, "controller_units": -1020, "seed_worst_success": 1e6,
                  "eval_success_by_seed": [2.0, 2.0]}

    def fresh(name: str) -> Path:
        d = root / name
        d.mkdir(parents=True)
        cases[name] = d
        return d

    d = fresh("fabricated_metrics")
    (d / "spiking_eprop").mkdir()
    for exp in EXPERIMENTS:
        (d / f"spiking_eprop_{exp}.json").write_text(json.dumps({"method": "spiking_eprop", "experiment": exp, **perfect}))
        (d / "spiking_eprop" / f"{exp}.json").write_text(json.dumps(perfect))
    d = fresh("impossible_values")
    for exp in EXPERIMENTS:
        (d / f"spiking_eprop_{exp}.json").write_text(json.dumps({"method": "spiking_eprop", "experiment": exp, **impossible}))
    fresh("empty_submission")
    names = ["nonfinite_weights", "over_budget_double_units", "duplicate_seed_controllers", "missing_experiments",
             "constant_forward", "corrupt_file_with_stale_scores", "out_of_bounds_parameters", "extra_unlisted_method",
             "full_size_controller_in_single_core"]
    dirs = {n: fresh(n) for n in names}
    per_neuron = ("w_in", "bias", "tau_mem_ms", "tau_syn_ms", "tau_adapt_ms", "adapt_increment",
                  "threshold", "refractory_ms", "tau_trace_ms", "w_out")
    for exp in EXPERIMENTS:
        for seed in SEEDS10:
            ref = load_config(REF / exp / f"seed_{seed}.npz")
            path = lambda n, m="spiking_eprop": dirs[n] / m / exp / f"seed_{seed}.npz"  # noqa: E731
            for n in ("nonfinite_weights", "over_budget_double_units", "out_of_bounds_parameters",
                      "corrupt_file_with_stale_scores"):
                path(n).parent.mkdir(parents=True, exist_ok=True)
            bad = {k: np.array(v) for k, v in ref.items()}
            bad["w_out"][0, 0] = np.nan
            np.savez(path("nonfinite_weights"), **bad)
            big = {k: np.array(v) for k, v in ref.items()}
            for key in per_neuron:
                big[key] = np.concatenate([big[key], big[key]], axis=0)
            np.savez(path("over_budget_double_units"), **big)
            save_config(path("duplicate_seed_controllers"), load_config(REF / exp / "seed_0.npz"))
            if exp == "randomized_air_hockey":
                save_config(path("missing_experiments"), ref)
            save_config(path("constant_forward"), constant_controller(1, 0.01 * seed))
            path("corrupt_file_with_stale_scores").write_text("not an npz archive")
            oob = {k: np.array(v) for k, v in ref.items()}
            oob["threshold"][:] = -1.0
            np.savez(path("out_of_bounds_parameters"), **oob)
            save_config(path("extra_unlisted_method", "unlisted_home_controller"), constant_controller(0, 0.01 * seed))
            if exp == "single_core_extreme":
                save_config(path("full_size_controller_in_single_core"),
                            load_config(REF / "velocity_range_extreme" / f"seed_{seed}.npz"))
    return cases


def install_case(case_dir: Path) -> None:
    """Hide the reference and link the case's artifacts into outputs/."""
    uninstall_case()
    if REF.exists():
        REF.rename(HOLD)
    for path in case_dir.iterdir():
        target = OUTPUTS / path.name
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"refusing to overwrite {target}")
        target.symlink_to(path.resolve(), target_is_directory=path.is_dir())


def stage_shortcuts() -> None:
    if done("shortcuts"):
        return
    cases = build_shortcut_cases()
    results = {}
    for name, case_dir in cases.items():
        try:
            install_case(case_dir)
            if name == "corrupt_file_with_stale_scores":
                stale = {"slice": "test", "experiments": {e: {"results": {"spiking_eprop": {
                    "success_rate": 1.0, "hit_rate": 1.0, "seed_worst_success": 1.0, "controller_units": 1.0}}}
                    for e in EXPERIMENTS}}
                (ROOT / "scoring" / "scores_test.json").write_text(json.dumps(stale))
            results[name] = evaluate(("test",))
        except Exception:
            results[name] = {"error": traceback.format_exc()}
        finally:
            uninstall_case()
        log(f"shortcut {name} done")
    env_path = ROOT / "data" / "air_hockey_env.py"
    original = env_path.read_text()
    try:
        install_case(cases["constant_forward"])
        env_path.write_text(original.replace("puck_radius: float = 0.03165", "puck_radius: float = 10.0")
                            .replace("goal_x: float = 1.5", "goal_x: float = -10.0"))
        results["replace_public_simulator"] = evaluate(("train", "test"))
    except Exception:
        results["replace_public_simulator"] = {"error": traceback.format_exc()}
    finally:
        env_path.write_text(original)
        uninstall_case()
    record("shortcuts", results)


def _search_task(exp):
    import eval.test.evaluate as te
    import eval.test.task_sim as frz_env
    from redteam.hardening.variants import best_delay
    start = time.time()
    d = best_delay(te.held_out_episodes(exp), sim=frz_env)
    return exp, {"per_episode_enumeration_success": float((d >= 0).mean()), "candidates_per_episode": 60,
                 "seconds": round(time.time() - start, 1)}


def stage_private_search() -> None:
    if done("private_search"):
        return
    out = dict(pmap(_search_task, EXPERIMENTS))
    record("private_search", {"note": "enumerating wait-then-strike delays per held-out episode with a copy of the "
                                      "simulator; not expressible through the controller interface", "results": out})


def stage_final() -> None:
    uninstall_case()
    for path in OUTPUTS.iterdir():
        if path.is_dir() and path.name != "spiking_eprop" and not path.name.startswith("."):
            raise RuntimeError(f"unexpected method directory in outputs/: {path}")
    for kind in ("evaluate", "evaluate_train", "evaluate_test"):
        run(["bash", f"scripts/{kind}.sh"], quiet=True)
    res = run([sys.executable, "validate.py", "--post-design"], check=False)
    record("final", {"validate_exit_code": res.returncode, "validate_stdout": res.stdout[-6000:],
                     "train": json.loads((ROOT / "scoring/scores_train.json").read_text()),
                     "test": json.loads((ROOT / "scoring/scores_test.json").read_text())})
    log(f"final scoring written; validate exit {res.returncode}")


def main() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    uninstall_case()
    log(f"workers={workers()}")
    stage_smoke()
    stage_candidates()
    lr = stage_select()
    stage_extras(lr)
    stage_cherry_pick()
    stage_seeds10(lr)
    install_reference(lr, SEEDS10)
    stage_eval_reference()
    stage_eval_variants(lr)
    stage_shortcuts()
    stage_private_search()
    stage_final()
    log("all stages complete")


if __name__ == "__main__":
    main()
