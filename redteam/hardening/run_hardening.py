#!/usr/bin/env python3
"""Hardening job: reproduce the reference, run variants and shortcuts through the
unmodified evaluators, and regenerate the final reference-only scores.

Every stage writes ``redteam/hardening/results/<stage>.json`` and is skipped
when that file exists, so a killed job resumes where it stopped.  Variant
controllers live in ``redteam/hardening/_staging`` (gitignored) and are linked
into ``outputs/`` only while being scored.  ``outputs/spiking_eprop`` is hidden
as ``outputs/_ref_hold`` (names starting with ``_`` are not scored) during the
shortcut cases and always restored.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path("/home/user")
HERE = ROOT / "redteam" / "hardening"
STAGING = HERE / "_staging"
RESULTS = HERE / "results"
OUTPUTS = ROOT / "outputs"
REF = OUTPUTS / "spiking_eprop"
HOLD = OUTPUTS / "_ref_hold"
CONDITIONS = ["random_position_speed", "speed_wide", "speed_extreme"]
EXPERIMENTS = {
    "random_position_speed": "randomized_air_hockey",
    "speed_wide": "velocity_range_wide",
    "speed_extreme": "velocity_range_extreme",
}
SEEDS = [0, 1, 2, 3, 4]
ENV = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"))
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


def record(stage: str, payload) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")


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


def run_pool(cmds: list[list[str]]) -> None:
    """Run commands with as many concurrent processes as CPUs."""
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
        raise RuntimeError(f"{len(failures)} training commands failed: {failures[:2]}")


def restore_reference() -> None:
    if HOLD.exists() and not REF.exists():
        HOLD.rename(REF)
    for path in OUTPUTS.iterdir() if OUTPUTS.exists() else []:
        if path.is_symlink():
            path.unlink()


def scores(kind: str) -> dict:
    return json.loads((ROOT / "scoring" / f"scores_{kind}.json").read_text())


def evaluate(kinds=("test",)) -> dict:
    out = {}
    for kind in kinds:
        start = time.time()
        res = run(["bash", f"scripts/evaluate_{kind}.sh"], check=False, quiet=True)
        out[kind] = {"exit_code": res.returncode, "seconds": round(time.time() - start, 1)}
        path = ROOT / "scoring" / f"scores_{kind}.json"
        out[kind]["scores"] = json.loads(path.read_text()) if path.exists() else None
        log(f"evaluate_{kind}.sh exit={res.returncode} in {out[kind]['seconds']}s")
    return out


def train_cmds(out_dir: Path, lr: float, episodes: int, extra: list[str] | None = None,
               snapshots: list[str] | None = None, seeds=SEEDS, conditions=CONDITIONS) -> list[list[str]]:
    cmds = []
    for cond in conditions:
        for seed in seeds:
            cmd = [sys.executable, "method/train.py", "--conditions", cond, "--seeds", str(seed),
                   "--episodes", str(episodes), "--learning-rate", repr(lr), "--output-dir", str(out_dir)]
            for snap in snapshots or []:
                cmd += ["--snapshot", snap]
            cmds.append(cmd + (extra or []))
    return cmds


# --------------------------------------------------------------------------- stages

def stage_smoke() -> None:
    if done("smoke"):
        return
    sys.path.insert(0, str(ROOT))
    import data.air_hockey_env as pub_env
    import data.neuromorphic_chip as pub_chip
    import eval.test.chip_runtime as frz_chip
    import eval.test.task_sim as frz_env
    from method.spiking_eprop_rl import EpropConfig, train_controller

    def body(path: Path) -> str:
        text = path.read_text()
        return text[text.index('"""', 3):]  # ignore the frozen-copy header line

    same_src = {
        "env": body(ROOT / "data/air_hockey_env.py")[3:] in (ROOT / "eval/test/task_sim.py").read_text(),
        "chip": body(ROOT / "data/neuromorphic_chip.py")[3:] in (ROOT / "eval/test/chip_runtime.py").read_text(),
    }
    rng = np.random.default_rng(0)
    timing = {}
    for name, cond in [("main", "random_position_speed"), ("extreme", "speed_extreme")]:
        init = pub_env.sample_initial_conditions(cond, rng, 64)
        acts = rng.integers(0, 2, size=(100, 64))
        a, b = pub_env.AirHockeyBatch(init), frz_env.AirHockeyBatch(init)
        for k in range(100):
            oa = a.step(acts[k])
            ob = b.step(acts[k])
        assert np.array_equal(oa[0], ob[0]) and np.array_equal(a.success, b.success)
        timing[f"random_success_{name}"] = float(a.success.mean())
    cfg = EpropConfig(hidden_neurons=64, cells_per_dim=4, calibration_episodes=4)
    start = time.time()
    config, hist = train_controller("random_position_speed", 0, 6, cfg)
    timing["tiny_train_s"] = time.time() - start
    chip_a, chip_b = pub_chip.ChipBatch(config, 8), frz_chip.ChipBatch(config, 8)
    obs = pub_env.AirHockeyBatch(pub_env.sample_initial_conditions("speed_wide", rng, 8)).observe()
    assert np.array_equal(chip_a.step(obs), chip_b.step(obs))
    # Real-size timing probes.
    big = EpropConfig()
    start = time.time()
    config, hist = train_controller("random_position_speed", 0, 20, big)
    timing["train_20_episodes_1020_units_s"] = time.time() - start
    timing["reservoir"] = hist["reservoir"]
    import eval.test.evaluate as te
    init = te.held_out_episodes("randomized_air_hockey")[:500]
    start = time.time()
    s, h = te.rollout_controller(frz_chip.validate_config(config), init)
    timing["eval_500_episodes_1020_units_s"] = time.time() - start
    timing["eval_success_untrained"] = float(s.mean())
    record("smoke", {"same_source": same_src, "timing": timing, "workers": workers(),
                     "numpy": np.__version__})
    log(f"smoke ok: {timing} same_source={same_src} workers={workers()}")
    if not all(same_src.values()):
        raise RuntimeError("frozen test copies differ from data/ sources")


def validation_success(config_dir: Path, condition: str, seeds, n: int = 300) -> float:
    sys.path.insert(0, str(ROOT))
    from data.air_hockey_env import AirHockeyBatch, sample_initial_conditions
    from data.neuromorphic_chip import ChipBatch, load_config
    rng = np.random.default_rng([777, CONDITIONS.index(condition)])
    init = sample_initial_conditions(condition, rng, n)
    vals = []
    for seed in seeds:
        config = load_config(config_dir / EXPERIMENTS[condition] / f"seed_{seed}.npz")
        chip = ChipBatch(config, n)
        env = AirHockeyBatch(init)
        rows = np.arange(n)
        obs = env.observe()
        while rows.size:
            act = np.zeros(n, dtype=np.int64)
            act[rows] = np.argmax(chip.scores(chip.step(obs[rows])), axis=1)
            obs, _, d, _ = env.step(act)
            keep = ~d[rows]
            if not np.all(keep):
                chip.keep(np.flatnonzero(keep))
                rows = rows[keep]
        vals.append(float(env.success.mean()))
    return float(np.mean(vals))


SWEEP = [0.002, 0.007, 0.025, 0.08]


def stage_sweep() -> float:
    if done("sweep"):
        return json.loads((RESULTS / "sweep.json").read_text())["chosen_learning_rate"]
    cmds = []
    for lr in SWEEP:
        cmds += train_cmds(STAGING / f"sweep_lr{lr}", lr, 2000, seeds=[100], conditions=["random_position_speed"])
    run_pool(cmds)
    result = {}
    for lr in SWEEP:
        d = STAGING / f"sweep_lr{lr}"
        logd = json.loads((d / "randomized_air_hockey" / "training_seed_100.json").read_text())
        result[str(lr)] = {
            "greedy_validation_success": validation_success(d, "random_position_speed", [100]),
            "online_last200": logd["online_success_last200"],
            "online_first200": logd["online_success_first200"],
            "wall_seconds": logd["wall_seconds"],
        }
    chosen = max(SWEEP, key=lambda lr: (result[str(lr)]["greedy_validation_success"], result[str(lr)]["online_last200"]))
    record("sweep", {"seed": 100, "condition": "random_position_speed", "episodes": 2000,
                     "results": result, "chosen_learning_rate": chosen,
                     "note": "selection on public validation episodes, never on the held-out test"})
    log(f"sweep: {result} -> {chosen}")
    return chosen


def stage_reference(lr: float) -> None:
    if done("reference_training"):
        return
    snaps = [f"0={STAGING / 'spiking_eprop_untrained'}", f"500={STAGING / 'spiking_eprop_500ep'}",
             f"2000={REF}"]
    run_pool(train_cmds(STAGING / "spiking_eprop_4000ep", lr, 4000, snapshots=snaps))
    # Reference training logs (first 2000 episodes of each uninterrupted run).
    summary = {}
    for cond in CONDITIONS:
        exp = EXPERIMENTS[cond]
        for seed in SEEDS:
            hist = json.loads((STAGING / "spiking_eprop_4000ep" / exp / f"training_seed_{seed}.json").read_text())
            online = np.array([int(c) for c in hist["online_success"][:2000]])
            ref_log = dict(hist)
            ref_log.update({"episodes": 2000, "online_success": hist["online_success"][:2000],
                            "online_hit": hist["online_hit"][:2000], "episode_steps": hist["episode_steps"][:2000],
                            "online_success_last200": float(online[-200:].mean()), "method": "spiking_eprop",
                            "note": "first 2000 episodes of the uninterrupted 4000-episode run"})
            ref_log.pop("wall_seconds", None)
            (REF / exp / f"training_seed_{seed}.json").write_text(json.dumps(ref_log) + "\n")
            curve = [float(online[i:i + 100].mean()) for i in range(0, 2000, 100)]
            summary.setdefault(exp, {})[f"seed_{seed}"] = {
                "online_success_per_100": curve,
                "online_last200_at_4000": hist["online_success_last200"],
                "wall_seconds_4000": hist["wall_seconds"],
                "reservoir": hist["reservoir"],
            }
    record("reference_training", {"learning_rate": lr, "summary": summary})
    log("reference training complete")


def stage_variants(lr: float) -> None:
    if done("variant_training"):
        return
    cmds = train_cmds(STAGING / "spiking_eprop_20cells", lr, 2000, extra=["--cells-per-dim", "20"])
    cmds += train_cmds(STAGING / "spiking_eprop_critic", lr, 2000, extra=["--critic-learning-rate", "0.02"])
    run_pool(cmds)
    record("variant_training", {"learning_rate": lr, "variants": {
        "spiking_eprop_20cells": "same rule and budget, 20 instead of 10 population cells per state variable",
        "spiking_eprop_critic": "same reservoir and budget, return minus a learned value readout (actor-critic e-prop)",
    }})
    log("variant training complete")


def stage_distill() -> None:
    if done("distill"):
        return
    sys.path.insert(0, str(ROOT))
    from data.neuromorphic_chip import load_config, save_config
    from redteam.hardening.variants import distill_readout
    for cond in CONDITIONS:
        exp = EXPERIMENTS[cond]
        for seed in SEEDS:
            out = STAGING / "oracle_distilled" / exp / f"seed_{seed}.npz"
            if out.exists():
                continue
            base = load_config(STAGING / "spiking_eprop_untrained" / exp / f"seed_{seed}.npz")
            save_config(out, distill_readout(base, cond, seed))
            log(f"distilled {exp} seed {seed}")
    record("distill", {"note": "reference reservoirs, readout fit to per-episode enumeration teacher (out of bounds)"})


def build_shortcut_cases() -> dict[str, Path]:
    sys.path.insert(0, str(ROOT))
    from data.neuromorphic_chip import load_config, save_config
    from redteam.hardening.variants import constant_controller
    root = STAGING / "shortcuts"
    cases = {}
    perfect = {"success_rate": 1.0, "hit_rate": 1.0, "controller_units": 1, "seed_worst_success": 1.0,
               "eval_success_by_seed": [1.0, 1.0, 1.0]}
    impossible = {"success_rate": 2.0, "hit_rate": 1e6, "controller_units": -1020, "seed_worst_success": 1e6,
                  "eval_success_by_seed": [2.0, float("nan")]}

    def fresh(name: str) -> Path:
        d = root / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        cases[name] = d
        return d

    d = fresh("fabricated_metrics")
    for cond in CONDITIONS:
        (d / f"spiking_eprop_{cond}.json").write_text(json.dumps({"method": "spiking_eprop", "condition": cond, **perfect}))
        (d / "spiking_eprop").mkdir(exist_ok=True)
        (d / "spiking_eprop" / f"{EXPERIMENTS[cond]}.json").write_text(json.dumps(perfect))
    d = fresh("impossible_values")
    for cond in CONDITIONS:
        (d / f"spiking_eprop_{cond}.json").write_text(json.dumps({"method": "spiking_eprop", "condition": cond, **impossible}))
    fresh("empty_submission")
    d = fresh("nonfinite_weights")
    d2 = fresh("over_budget_2040_units")
    d3 = fresh("duplicate_seed_controllers")
    d4 = fresh("missing_experiments")
    d5 = fresh("constant_forward")
    d6 = fresh("corrupt_file_with_stale_scores")
    d7 = fresh("out_of_bounds_parameters")
    d8 = fresh("extra_unlisted_method")
    for cond in CONDITIONS:
        exp = EXPERIMENTS[cond]
        for seed in SEEDS:
            ref = load_config(REF / exp / f"seed_{seed}.npz")
            bad = {k: np.array(v) for k, v in ref.items()}
            bad["w_out"][0, 0] = np.nan
            (d / "spiking_eprop" / exp).mkdir(parents=True, exist_ok=True)
            np.savez(d / "spiking_eprop" / exp / f"seed_{seed}.npz", **bad)
            big = {k: np.array(v) for k, v in ref.items()}
            for key in ("w_in", "bias", "tau_mem_ms", "tau_syn_ms", "tau_adapt_ms", "adapt_increment",
                        "threshold", "refractory_ms", "tau_trace_ms", "w_out"):
                big[key] = np.concatenate([big[key], big[key]], axis=0)
            (d2 / "spiking_eprop" / exp).mkdir(parents=True, exist_ok=True)
            np.savez(d2 / "spiking_eprop" / exp / f"seed_{seed}.npz", **big)
            save_config(d3 / "spiking_eprop" / exp / f"seed_{seed}.npz", load_config(REF / exp / "seed_0.npz"))
            if cond == "random_position_speed":
                save_config(d4 / "spiking_eprop" / exp / f"seed_{seed}.npz", ref)
            save_config(d5 / "spiking_eprop" / exp / f"seed_{seed}.npz", constant_controller(1, 0.01 * seed))
            (d6 / "spiking_eprop" / exp).mkdir(parents=True, exist_ok=True)
            (d6 / "spiking_eprop" / exp / f"seed_{seed}.npz").write_text("not an npz archive")
            oob = {k: np.array(v) for k, v in ref.items()}
            oob["threshold"][:] = -1.0
            (d7 / "spiking_eprop" / exp).mkdir(parents=True, exist_ok=True)
            np.savez(d7 / "spiking_eprop" / exp / f"seed_{seed}.npz", **oob)
            save_config(d8 / "unlisted_home_controller" / exp / f"seed_{seed}.npz", constant_controller(0, 0.01 * seed))
    return cases


def install_case(case_dir: Path) -> None:
    """Hide the reference and link the case's artifacts into outputs/."""
    restore_reference()
    if REF.exists():
        REF.rename(HOLD)
    for path in case_dir.iterdir():
        target = OUTPUTS / path.name
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"refusing to overwrite {target}")
        target.symlink_to(path.resolve(), target_is_directory=path.is_dir())


def uninstall_case() -> None:
    for path in list(OUTPUTS.iterdir()):
        if path.is_symlink():
            path.unlink()
    restore_reference()


def stage_eval_reference() -> None:
    if done("eval_reference"):
        return
    restore_reference()
    res = evaluate(("train", "test"))
    record("eval_reference", res)


VARIANTS = ["spiking_eprop_untrained", "spiking_eprop_500ep", "spiking_eprop_4000ep",
            "spiking_eprop_20cells", "spiking_eprop_critic", "oracle_distilled"]


def stage_eval_variants() -> None:
    if done("eval_variants"):
        return
    restore_reference()
    for name in VARIANTS:
        (OUTPUTS / name).symlink_to((STAGING / name).resolve(), target_is_directory=True)
    try:
        res = evaluate(("test",))
    finally:
        uninstall_case()
    record("eval_variants", res)


def stage_shortcuts() -> None:
    if done("shortcuts"):
        return
    cases = build_shortcut_cases()
    results = {}
    for name, case_dir in cases.items():
        install_case(case_dir)
        try:
            if name == "corrupt_file_with_stale_scores":
                stale = {"slice": "test", "experiments": {e: {"results": {"spiking_eprop": {
                    "success_rate": 1.0, "hit_rate": 1.0, "seed_worst_success": 1.0, "controller_units": 1.0}}}
                    for e in EXPERIMENTS.values()}}
                (ROOT / "scoring" / "scores_test.json").write_text(json.dumps(stale))
            results[name] = evaluate(("test",))
        except Exception:
            results[name] = {"error": traceback.format_exc()}
        finally:
            uninstall_case()
        log(f"shortcut {name} done")
    # Replacing the public simulator only changes the visible slice.
    env_path = ROOT / "data" / "air_hockey_env.py"
    original = env_path.read_text()
    install_case(cases["constant_forward"])
    try:
        env_path.write_text(original.replace("puck_radius: float = 0.03165", "puck_radius: float = 10.0")
                            .replace("goal_x: float = 1.5", "goal_x: float = -10.0"))
        results["replace_public_simulator"] = evaluate(("train", "test"))
    finally:
        env_path.write_text(original)
        uninstall_case()
    record("shortcuts", results)


def stage_private_search() -> None:
    if done("private_search"):
        return
    sys.path.insert(0, str(ROOT))
    import eval.test.evaluate as te
    import eval.test.task_sim as frz_env
    from redteam.hardening.variants import best_delay
    out = {}
    for exp in te.EXPERIMENTS:
        init = te.held_out_episodes(exp)
        start = time.time()
        d = best_delay(init, sim=frz_env)
        out[exp] = {"per_episode_enumeration_success": float((d >= 0).mean()),
                    "candidates_per_episode": 60, "seconds": round(time.time() - start, 1)}
        log(f"private search {exp}: {out[exp]}")
    record("private_search", {"note": "enumerating wait-then-strike delays per held-out episode with a copy of "
                                      "the simulator; impossible through the controller interface", "results": out})


def stage_final() -> None:
    restore_reference()
    for path in OUTPUTS.iterdir():
        if path.is_dir() and path.name != "spiking_eprop" and not path.name.startswith("."):
            raise RuntimeError(f"unexpected method directory in outputs/: {path}")
    for kind in ("evaluate", "evaluate_train", "evaluate_test"):
        run(["bash", f"scripts/{kind}.sh"])
    res = run([sys.executable, "validate.py", "--post-design"], check=False)
    record("final", {"validate_exit_code": res.returncode, "validate_stdout": res.stdout[-6000:],
                     "train": scores("train"), "test": scores("test")})
    log(f"final scoring written; validate exit {res.returncode}")


def main() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    restore_reference()
    uninstall_case()
    log(f"workers={workers()}")
    stage_smoke()
    lr = stage_sweep()
    stage_reference(lr)
    stage_eval_reference()
    stage_variants(lr)
    stage_distill()
    stage_eval_variants()
    stage_shortcuts()
    stage_private_search()
    stage_final()
    log("all stages complete")


if __name__ == "__main__":
    main()
