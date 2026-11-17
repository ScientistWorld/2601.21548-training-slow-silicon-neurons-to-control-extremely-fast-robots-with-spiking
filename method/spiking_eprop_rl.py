"""Reproduction of the paper's spiking reservoir controller with e-prop readout learning.

Paper, Sec. II-B: 6 state variables -> Gaussian population code (10 cells per
variable, 60 deterministic spike generators) -> 1020 adaptive LIF silicon
neurons with *fixed random feedforward* synapses -> 2 linear readout units on
the filtered hidden spike trains -> softmax action probabilities.  Each control
step runs the analog dynamics for 20 ms.  Only the hidden-to-readout weights
are plastic and are updated every two trials with Eq. 1:

    dw_ik = -alpha * sum_t r^t * sum_{t'<=t} gamma^(t-t') (pi_k^t' - 1[a^t'=k]) sbar_i^t'

The silicon substrate is replaced by the software runtime in
``data/neuromorphic_chip.py`` (1 ms LIF dynamics with synaptic filtering,
spike-frequency adaptation, refractoriness and device mismatch drawn per
reservoir seed).  The trained controller is exported in that runtime's
configuration format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
from typing import Callable

import numpy as np

from data.air_hockey_env import CONDITIONS, AirHockeyBatch, sample_initial_conditions
from data.neuromorphic_chip import ChipBatch, validate_config, load_config


@dataclass
class EpropConfig:
    hidden_neurons: int = 1020
    cells_per_dim: int = 10
    input_rate_hz: float = 200.0
    connection_prob: float = 0.2
    inhibitory_fraction: float = 0.2
    target_active_fraction: float = 0.25
    tau_mem_ms: float = 20.0
    tau_syn_ms: float = 5.0
    tau_adapt_ms: float = 100.0
    adapt_increment: float = 0.002
    refractory_ms: float = 2.0
    tau_trace_ms: float = 30.0
    mismatch_cv: float = 0.2
    # alpha of Eq. 1, expressed in units of 1 / E[||sbar||^2] of the untrained
    # reservoir so one value transfers across reservoir samples.
    learning_rate: float = 0.012
    gamma: float = 0.95
    update_every_trials: int = 2
    init_logit_std: float = 0.1
    # Variant switch (NOT the paper's rule): subtract a learned value readout
    # from the return (actor-critic e-prop).  Off for the reproduction.
    critic_learning_rate: float = 0.0
    calibration_episodes: int = 16
    extra: dict = field(default_factory=dict)


def encoder_ranges(condition: str) -> np.ndarray:
    """Population-code range per state variable (adaptive to the condition).

    The velocity code spans the condition's speed range (both directions,
    since the puck bounces off the side walls), so wider speed ranges are
    represented with coarser resolution by the same 10 cells (Sec. III-B).
    """
    vmax = max(0.2, float(CONDITIONS[condition]["speed"][1]))
    return np.array([
        [0.80, 1.60],    # x_p
        [-0.50, 0.50],   # y_p
        [-1.00, 4.00],   # v_x
        [-vmax, vmax],   # v_y
        [0.70, 1.50],    # x_ee
        [-0.05, 0.05],   # y_ee
    ])


def _lognormal(rng: np.random.Generator, cv: float, size) -> np.ndarray:
    return rng.lognormal(mean=0.0, sigma=cv, size=size)


def _random_rollout_states(condition: str, rng: np.random.Generator, episodes: int) -> np.ndarray:
    env = AirHockeyBatch(sample_initial_conditions(condition, rng, episodes))
    states = [env.observe()[~env.done]]
    while not np.all(env.done):
        obs, _, _, _ = env.step(rng.integers(0, 2, size=episodes))
        states.append(obs[~env.done])
    return np.concatenate(states, axis=0)


def build_reservoir(seed: int, condition: str, cfg: EpropConfig) -> tuple[dict[str, np.ndarray], dict]:
    """Sample a fixed random reservoir ("chip instance") and an untrained readout."""
    frozen = Path(__file__).resolve().parents[1] / "data" / "backbones" / f"{condition}_{cfg.hidden_neurons}" / f"seed_{seed}.npz"
    if frozen.exists() and not cfg.extra.get("generate_backbone", False):
        return load_config(frozen), json.loads(frozen.with_suffix(".json").read_text())
    rng = np.random.default_rng([seed, 7919])
    k = cfg.cells_per_dim
    n = cfg.hidden_neurons
    ranges = encoder_ranges(condition)
    centers = np.stack([np.linspace(lo, hi, k) for lo, hi in ranges])
    spacing = (ranges[:, 1] - ranges[:, 0]) / max(1, k - 1)
    sigma = np.repeat(spacing[:, None], k, axis=1)
    rates = np.full((6, k), cfg.input_rate_hz)

    n_in = 6 * k
    mask = rng.random((n, n_in)) < cfg.connection_prob
    sign = np.where(rng.random((n, n_in)) < cfg.inhibitory_fraction, -1.0, 1.0)
    w_shape = mask * sign * _lognormal(rng, 0.25, (n, n_in))

    tau_mem = cfg.tau_mem_ms * _lognormal(rng, cfg.mismatch_cv, n)
    tau_syn = cfg.tau_syn_ms * _lognormal(rng, cfg.mismatch_cv, n)
    tau_adapt = cfg.tau_adapt_ms * _lognormal(rng, cfg.mismatch_cv, n)
    threshold = _lognormal(rng, 0.5 * cfg.mismatch_cv, n)
    adapt_inc = cfg.adapt_increment * _lognormal(rng, cfg.mismatch_cv, n)

    # Bias calibration of the chip: one global synaptic gain chosen so that a
    # target fraction of (neuron, state) pairs is driven above threshold in
    # the steady state over states visited by random behaviour.
    calib_rng = np.random.default_rng([seed, 104729])
    states = _random_rollout_states(condition, calib_rng, cfg.calibration_episodes)
    if states.shape[0] > 600:
        states = states[calib_rng.choice(states.shape[0], 600, replace=False)]
    dim_index = np.repeat(np.arange(6), k)
    z = (states[:, dim_index] - centers.reshape(-1)) / sigma.reshape(-1)
    in_rate = rates.reshape(-1) * 1e-3 * np.exp(-0.5 * z * z)          # spikes per ms
    a_mem = np.exp(-1.0 / tau_mem)
    a_syn = np.exp(-1.0 / tau_syn)
    drive = (in_rate @ w_shape.T) / ((1.0 - a_syn) * (1.0 - a_mem))    # [S, N]
    ratio = np.where(drive > 0, threshold / np.maximum(drive, 1e-12), np.inf)
    gain = float(np.quantile(ratio, cfg.target_active_fraction))
    if not np.isfinite(gain):
        gain = float(np.quantile(ratio[np.isfinite(ratio)], 0.5))

    config = {
        "enc_centers": centers,
        "enc_sigma": sigma,
        "enc_max_rate_hz": rates,
        "w_in": gain * w_shape,
        "bias": np.zeros(n),
        "tau_mem_ms": tau_mem,
        "tau_syn_ms": tau_syn,
        "tau_adapt_ms": tau_adapt,
        "adapt_increment": adapt_inc,
        "threshold": threshold,
        "refractory_ms": np.full(n, cfg.refractory_ms),
        "tau_trace_ms": np.full(n, cfg.tau_trace_ms),
        "w_out": np.zeros((n, 2)),
        "b_out": np.zeros(2),
    }
    config = validate_config(config)

    # Readout input power of the untrained reservoir under random behaviour.
    chip = ChipBatch(config, cfg.calibration_episodes)
    env = AirHockeyBatch(sample_initial_conditions(condition, calib_rng, cfg.calibration_episodes))
    rows = np.arange(cfg.calibration_episodes)
    obs = env.observe()
    power, rate_sum, count = 0.0, 0.0, 0
    while rows.size:
        trace = chip.step(obs[rows]).astype(np.float64)
        power += float(np.sum(trace * trace))
        rate_sum += float(np.sum(chip.spike_count))
        count += rows.size
        act = np.zeros(env.batch, dtype=np.int64)
        act[rows] = calib_rng.integers(0, 2, size=rows.size)
        obs, _, done, _ = env.step(act)
        keep = ~done[rows]
        if not np.all(keep):
            chip.keep(np.flatnonzero(keep))
            rows = rows[keep]
    trace_power = power / max(1, count)
    stats = {
        "gain": gain,
        "trace_power": trace_power,
        "mean_rate_hz": rate_sum / max(1, count * n) / 0.020,
        "active_fraction_target": cfg.target_active_fraction,
    }
    w_rng = np.random.default_rng([seed, 15485863])
    w_out = w_rng.normal(0.0, cfg.init_logit_std / np.sqrt(max(trace_power, 1e-9)), (n, 2))
    config["w_out"] = w_out.astype(np.float32)
    return config, stats


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class EpropReadoutLearner:
    """Online e-prop readout learning on a fixed reservoir (Eq. 1)."""

    def __init__(self, config: dict[str, np.ndarray], trace_power: float, cfg: EpropConfig, seed: int) -> None:
        self.config = dict(config)
        self.cfg = cfg
        self.w_out = np.asarray(config["w_out"], dtype=np.float64).copy()
        self.b_out = np.asarray(config["b_out"], dtype=np.float64).copy()
        self.alpha = cfg.learning_rate / max(trace_power, 1e-9)
        self.beta = cfg.critic_learning_rate / max(trace_power, 1e-9)
        self.w_value = np.zeros(self.w_out.shape[0])
        self.chip = ChipBatch(self.config, cfg.update_every_trials)
        self.policy_rng = np.random.default_rng([seed, 31337])

    def export(self) -> dict[str, np.ndarray]:
        out = dict(self.config)
        out["w_out"] = self.w_out.astype(np.float32)
        out["b_out"] = self.b_out.astype(np.float32)
        return validate_config(out)

    def run_trials(self, init: np.ndarray) -> tuple[list[dict], dict]:
        """Run ``len(init)`` trials in parallel with the current readout."""
        n_trials = init.shape[0]
        env = AirHockeyBatch(init)
        self.chip.reset(n_trials)
        rows = np.arange(n_trials)
        obs = env.observe()
        traj = [{"trace": [], "probs": [], "action": [], "reward": []} for _ in range(n_trials)]
        while rows.size:
            trace = self.chip.step(obs[rows]).astype(np.float64)
            probs = _softmax(trace @ self.w_out + self.b_out)
            actions = (self.policy_rng.random(rows.size) < probs[:, 1]).astype(np.int64)
            full = np.zeros(n_trials, dtype=np.int64)
            full[rows] = actions
            obs, reward, done, info = env.step(full)
            for j, r in enumerate(rows):
                traj[r]["trace"].append(trace[j].copy())
                traj[r]["probs"].append(probs[j])
                traj[r]["action"].append(actions[j])
                traj[r]["reward"].append(reward[r])
            keep = ~done[rows]
            if not np.all(keep):
                self.chip.keep(np.flatnonzero(keep))
                rows = rows[keep]
        return traj, info

    def update(self, trajectories: list[dict]) -> None:
        """Apply Eq. 1 summed over the trials since the last update.

        sum_t r^t sum_{t'<=t} gamma^(t-t') x^t' = sum_t' G^t' x^t' with the
        discounted return-to-go G^t' = sum_{t>=t'} gamma^(t-t') r^t, so the
        update below is Eq. 1 evaluated exactly.
        """
        grad = np.zeros_like(self.w_out)
        value_grad = np.zeros_like(self.w_value)
        n_steps = 0
        for tr in trajectories:
            s = np.asarray(tr["trace"])
            p = np.asarray(tr["probs"])
            a = np.asarray(tr["action"], dtype=np.int64)
            r = np.asarray(tr["reward"], dtype=np.float64)
            g = np.zeros_like(r)
            run = 0.0
            for t in range(len(r) - 1, -1, -1):
                run = r[t] + self.cfg.gamma * run
                g[t] = run
            signal = g
            if self.beta > 0.0:
                value = s @ self.w_value
                signal = g - value
                value_grad += s.T @ (g - value)
            n_steps += len(r)
            err = p.copy()
            err[np.arange(len(a)), a] -= 1.0
            grad += s.T @ (signal[:, None] * err)
        self.w_out -= self.alpha * grad
        if self.beta > 0.0:
            # Normalised LMS step for the value readout (variant only).
            self.w_value += self.beta * value_grad / max(1, n_steps)


def train_controller(
    condition: str,
    seed: int,
    episodes: int,
    cfg: EpropConfig,
    checkpoints: tuple[int, ...] = (),
    on_checkpoint: Callable[[int, dict[str, np.ndarray]], None] | None = None,
    log: Callable[[str], None] | None = None,
    resume_path: Path | None = None,
    checkpoint_every_trials: int = 500,
) -> tuple[dict[str, np.ndarray], dict]:
    """Train one reservoir seed online for ``episodes`` trials."""
    config, stats = build_reservoir(seed, condition, cfg)
    learner = EpropReadoutLearner(config, stats["trace_power"], cfg, seed)
    env_rng = np.random.default_rng([seed, 2601, 21548])
    successes: list[int] = []
    hits: list[int] = []
    lengths: list[int] = []
    pair = cfg.update_every_trials
    done_episodes = 0
    average = np.zeros_like(learner.w_out)
    average_count = 0
    # A checkpoint is taken only between complete two-trial updates. The chip
    # resets at each pair already, so these arrays and RNG states completely
    # determine continuation. No elapsed-time limit or load-dependent decision.
    digest = hashlib.sha256(json.dumps({
        "condition": condition, "seed": seed, "episodes": episodes,
        "config": vars(cfg)}, sort_keys=True).encode())
    root = Path(__file__).resolve().parents[1]
    for relative in ["method/spiking_eprop_rl.py", "method/train.py",
                     "data/air_hockey_env.py", "data/neuromorphic_chip.py",
                     "data/robot/air_hockey.xml"]:
        digest.update((root / relative).read_bytes())
    for key, value in sorted(config.items()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(json.dumps(stats,sort_keys=True).encode())
    fingerprint = digest.hexdigest()
    if resume_path is not None and resume_path.exists():
        with np.load(resume_path, allow_pickle=False) as saved:
            meta = json.loads(saved["metadata"].tobytes())
            if meta["fingerprint"] != fingerprint:
                raise ValueError("training checkpoint belongs to different code/configuration")
            learner.w_out = saved["w_out"].copy()
            learner.w_value = saved["w_value"].copy()
            average = saved["average"].copy()
            successes = saved["success"].astype(int).tolist()
            hits = saved["hit"].astype(int).tolist()
            lengths = saved["lengths"].astype(int).tolist()
        done_episodes = len(successes)
        average_count = meta["average_count"]
        learner.policy_rng.bit_generator.state = meta["policy_rng"]
        env_rng.bit_generator.state = meta["env_rng"]
        if log is not None: log(f"resume {condition} seed={seed} episodes={done_episodes}")

    def save_progress():
        if resume_path is None: return
        meta = {"fingerprint": fingerprint, "average_count": average_count,
                "policy_rng": learner.policy_rng.bit_generator.state,
                "env_rng": env_rng.bit_generator.state}
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = resume_path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, w_out=learner.w_out, w_value=learner.w_value,
                                average=average, success=np.array(successes,dtype=np.uint8),
                                hit=np.array(hits,dtype=np.uint8),lengths=np.array(lengths,dtype=np.int16),
                                metadata=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8))
        temporary.replace(resume_path)

    if done_episodes == 0 and on_checkpoint is not None and 0 in checkpoints:
        on_checkpoint(0, learner.export())
    while done_episodes < episodes:
        n = min(pair, episodes - done_episodes)
        init = sample_initial_conditions(condition, env_rng, n)
        traj, info = learner.run_trials(init)
        learner.update(traj)
        successes.extend(int(x) for x in info["success"])
        hits.extend(int(x) for x in info["hit"])
        lengths.extend(int(x) for x in info["steps"])
        prev = done_episodes
        done_episodes += n
        if cfg.extra.get("average_after", 0) and done_episodes > cfg.extra["average_after"]:
            average += learner.w_out
            average_count += 1
        for c in checkpoints:
            if prev < c <= done_episodes and c > 0 and on_checkpoint is not None:
                on_checkpoint(c, learner.export())
        if (done_episodes // checkpoint_every_trials) != (prev // checkpoint_every_trials):
            save_progress()
            if log is not None:
                window = successes[-200:]
                log(f"{condition} seed={seed} episodes={done_episodes} online_success(last200)={np.mean(window):.3f}")
    save_progress()
    history = {
        "online_success": "".join(str(s) for s in successes),
        "online_hit": "".join(str(h) for h in hits),
        "episode_steps": lengths,
        "reservoir": stats,
    }
    if average_count:
        learner.w_out = average / average_count
        history["averaged_updates"] = average_count
    return learner.export(), history
