"""Runtime for neuromorphic interception controllers (public problem asset).

In the paper the controller is fixed silicon (DYNAP-SE) driven by FPGA spike
generators, with a readout computed on the host.  Here a controller is a
*configuration file* (``.npz``) that fully specifies such a spiking network;
this module executes it.  Every evaluator runs submitted controllers through
an identical runtime, so a submission is judged by what its network does on
held-out episodes, and its size is read from the configuration itself.

Network executed for every 20 ms control step (1 ms resolution):

* input: one deterministic spike generator per (state dimension, cell).  The
  generator rate is ``max_rate * exp(-0.5 * ((x_d - center) / sigma)^2)`` for
  the current observation ``x``; spikes are emitted when an accumulating phase
  crosses 1 (regular, replayable spike trains);
* hidden: ``N`` leaky integrate-and-fire units with synaptic low-pass filter,
  subtractive spike-frequency adaptation, absolute refractory period and an
  optional recurrent weight matrix;
* readout: filtered hidden spike trains (``trace``) times ``w_out`` plus
  ``b_out`` give two action scores; evaluators pick the larger score.

Configuration keys (all numeric arrays; N hidden units, K cells per input dim)::

    enc_centers [6, K]  enc_sigma [6, K]  enc_max_rate_hz [6, K]
    w_in [N, 6K]        w_rec [N, N] (optional)
    bias, tau_mem_ms, tau_syn_ms, tau_adapt_ms, adapt_increment,
    threshold, refractory_ms, tau_trace_ms                      [N] each
    w_out [N, 2]        b_out [2]

Unknown keys are ignored.  ``validate_config`` enforces shapes, finiteness and
physical bounds; ``controller_units`` is the number of hidden units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

STATE_DIM = 6
N_ACTIONS = 2
SUBSTEPS = 20
MAX_HIDDEN_UNITS = 1020
MAX_CELLS_PER_DIM = 32

PER_NEURON_BOUNDS = {
    "bias": (-1e4, 1e4),
    "tau_mem_ms": (1.0, 1000.0),
    "tau_syn_ms": (0.0, 1000.0),
    "tau_adapt_ms": (1.0, 10000.0),
    "adapt_increment": (0.0, 1e4),
    "threshold": (1e-6, 1e6),
    "refractory_ms": (0.0, 50.0),
    "tau_trace_ms": (1.0, 1000.0),
}
WEIGHT_BOUND = 1e4
READOUT_BOUND = 1e6


class ConfigError(ValueError):
    """Raised when a controller configuration is malformed or out of bounds."""


def _array(raw: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in raw:
        raise ConfigError(f"missing key {key!r}")
    arr = np.asarray(raw[key])
    if arr.dtype.kind not in "fiu":
        raise ConfigError(f"{key} must be numeric, got dtype {arr.dtype}")
    arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ConfigError(f"{key} contains non-finite values")
    return arr


def _bounded(arr: np.ndarray, key: str, lo: float, hi: float) -> None:
    if arr.size and (arr.min() < lo or arr.max() > hi):
        raise ConfigError(f"{key} outside [{lo}, {hi}]")


def validate_config(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Check a raw configuration and return it as float32 arrays."""
    centers = _array(raw, "enc_centers")
    if centers.ndim != 2 or centers.shape[0] != STATE_DIM:
        raise ConfigError("enc_centers must have shape [6, K]")
    k = centers.shape[1]
    if not 1 <= k <= MAX_CELLS_PER_DIM:
        raise ConfigError(f"cells per dimension must be in [1, {MAX_CELLS_PER_DIM}]")
    n_inputs = STATE_DIM * k
    sigma = _array(raw, "enc_sigma")
    rate = _array(raw, "enc_max_rate_hz")
    for key, arr in (("enc_sigma", sigma), ("enc_max_rate_hz", rate)):
        if arr.shape != centers.shape:
            raise ConfigError(f"{key} must have shape [6, K]")
    _bounded(centers, "enc_centers", -1e3, 1e3)
    _bounded(sigma, "enc_sigma", 1e-6, 1e3)
    _bounded(rate, "enc_max_rate_hz", 0.0, 1000.0)

    w_in = _array(raw, "w_in")
    if w_in.ndim != 2 or w_in.shape[1] != n_inputs:
        raise ConfigError(f"w_in must have shape [N, {n_inputs}]")
    n = w_in.shape[0]
    if n < 1:
        raise ConfigError("at least one hidden unit is required")
    if n > MAX_HIDDEN_UNITS:
        raise ConfigError(f"{n} hidden units exceed the {MAX_HIDDEN_UNITS}-unit budget")
    _bounded(w_in, "w_in", -WEIGHT_BOUND, WEIGHT_BOUND)

    out: dict[str, np.ndarray] = {
        "enc_centers": centers, "enc_sigma": sigma, "enc_max_rate_hz": rate, "w_in": w_in,
    }
    if "w_rec" in raw:
        w_rec = _array(raw, "w_rec")
        if w_rec.shape != (n, n):
            raise ConfigError(f"w_rec must have shape [{n}, {n}]")
        _bounded(w_rec, "w_rec", -WEIGHT_BOUND, WEIGHT_BOUND)
        out["w_rec"] = w_rec
    for key, (lo, hi) in PER_NEURON_BOUNDS.items():
        arr = _array(raw, key)
        if arr.shape != (n,):
            raise ConfigError(f"{key} must have shape [{n}]")
        _bounded(arr, key, lo, hi)
        out[key] = arr
    w_out = _array(raw, "w_out")
    b_out = _array(raw, "b_out")
    if w_out.shape != (n, N_ACTIONS) or b_out.shape != (N_ACTIONS,):
        raise ConfigError("w_out must be [N, 2] and b_out [2]")
    _bounded(w_out, "w_out", -READOUT_BOUND, READOUT_BOUND)
    _bounded(b_out, "b_out", -READOUT_BOUND, READOUT_BOUND)
    out["w_out"] = w_out
    out["b_out"] = b_out
    return {key: value.astype(np.float32) for key, value in out.items()}


def load_config(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a controller configuration from an ``.npz`` file."""
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            raw = {key: np.array(data[key]) for key in data.files}
    except ConfigError:
        raise
    except Exception as exc:  # unreadable or not an npz archive
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    return validate_config(raw)


def save_config(path: str | Path, config: Mapping[str, np.ndarray]) -> None:
    clean = validate_config(config)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(path), **clean)


def controller_units(config: Mapping[str, np.ndarray]) -> int:
    return int(np.asarray(config["w_in"]).shape[0])


def _decay(tau_ms: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau_ms, dtype=np.float64)
    out = np.zeros_like(tau)
    pos = tau > 0
    out[pos] = np.exp(-1.0 / tau[pos])
    return out.astype(np.float32)


class ChipBatch:
    """Execute one validated configuration on a batch of episodes."""

    def __init__(self, config: Mapping[str, np.ndarray], batch: int) -> None:
        c = config
        self.n = int(c["w_in"].shape[0])
        k = int(c["enc_centers"].shape[1])
        self.dim_index = np.repeat(np.arange(STATE_DIM), k)
        self.centers = c["enc_centers"].reshape(-1).astype(np.float32)
        self.inv_sigma = (1.0 / c["enc_sigma"].reshape(-1)).astype(np.float32)
        self.rate_per_ms = (c["enc_max_rate_hz"].reshape(-1) * 1e-3).astype(np.float32)
        self.w_in_t = np.ascontiguousarray(c["w_in"].T, dtype=np.float32)
        w_rec = c.get("w_rec")
        self.w_rec_t = None
        if w_rec is not None and np.any(w_rec != 0):
            self.w_rec_t = np.ascontiguousarray(np.asarray(w_rec).T, dtype=np.float32)
        self.a_mem = _decay(c["tau_mem_ms"])
        self.a_syn = _decay(c["tau_syn_ms"])
        self.a_adapt = _decay(c["tau_adapt_ms"])
        self.a_trace = _decay(c["tau_trace_ms"])
        self.bias = np.asarray(c["bias"], dtype=np.float32)
        self.threshold = np.asarray(c["threshold"], dtype=np.float32)
        self.adapt_inc = np.asarray(c["adapt_increment"], dtype=np.float32)
        self.use_adapt = bool(np.any(self.adapt_inc > 0))
        self.ref_steps = np.rint(np.asarray(c["refractory_ms"])).astype(np.float32)
        self.use_ref = bool(np.any(self.ref_steps > 0))
        self.w_out = np.asarray(c["w_out"], dtype=np.float32)
        self.b_out = np.asarray(c["b_out"], dtype=np.float32)
        self.reset(batch)

    def reset(self, batch: int) -> None:
        n_in = self.centers.shape[0]
        self.batch = int(batch)
        self.phase = np.zeros((batch, n_in), dtype=np.float32)
        self.i_syn = np.zeros((batch, self.n), dtype=np.float32)
        self.v = np.zeros((batch, self.n), dtype=np.float32)
        self.adapt = np.zeros((batch, self.n), dtype=np.float32)
        self.refractory = np.zeros((batch, self.n), dtype=np.float32)
        self.trace = np.zeros((batch, self.n), dtype=np.float32)
        self.spikes = np.zeros((batch, self.n), dtype=np.float32)
        self.spike_count = np.zeros((batch, self.n), dtype=np.float32)

    def keep(self, rows: np.ndarray) -> None:
        """Restrict the batch to ``rows`` (used when episodes finish)."""
        for name in ("phase", "i_syn", "v", "adapt", "refractory", "trace", "spikes", "spike_count"):
            setattr(self, name, getattr(self, name)[rows])
        self.batch = int(len(rows))

    def input_rates(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)[:, self.dim_index]
        z = (x - self.centers) * self.inv_sigma
        return self.rate_per_ms * np.exp(-0.5 * z * z)

    def step(self, obs: np.ndarray) -> np.ndarray:
        """Run 20 ms of network dynamics driven by ``obs`` [B, 6]; return traces."""
        rate = self.input_rates(obs)
        phase, i_syn, v, trace = self.phase, self.i_syn, self.v, self.trace
        spikes = self.spikes
        self.spike_count[:] = 0.0
        for _ in range(SUBSTEPS):
            phase += rate
            s_in = (phase >= 1.0).astype(np.float32)
            phase -= s_in
            i_syn *= self.a_syn
            i_syn += s_in @ self.w_in_t
            if self.w_rec_t is not None:
                i_syn += spikes @ self.w_rec_t
            v *= self.a_mem
            v += i_syn
            v += self.bias
            if self.use_adapt:
                v -= self.adapt
            if self.use_ref:
                v *= self.refractory <= 0.0
            np.greater_equal(v, self.threshold, out=spikes, casting="unsafe")
            v *= 1.0 - spikes
            if self.use_adapt:
                self.adapt *= self.a_adapt
                self.adapt += self.adapt_inc * spikes
            if self.use_ref:
                self.refractory = np.where(spikes > 0, self.ref_steps, np.maximum(self.refractory - 1.0, 0.0))
            trace *= self.a_trace
            trace += spikes
            self.spike_count += spikes
        return trace

    def scores(self, trace: np.ndarray) -> np.ndarray:
        return trace @ self.w_out + self.b_out
