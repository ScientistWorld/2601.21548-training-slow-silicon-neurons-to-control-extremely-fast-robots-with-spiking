"""Software substitute for the paper's fixed spiking substrate (public asset).

Paper, Sec. II-B: each of the six state variables is encoded by 10 Gaussian
population cells (60 deterministic spike generators); a fixed random
feedforward layer of 1020 adaptive leaky integrate-and-fire neurons processes
them for 20 ms per control step; the readout sees the *filtered* hidden spike
trains. Here a "chip" is a dictionary of numeric arrays that fully specifies one
such substrate, and ``ChipBank`` runs many chips in lockstep at 1 ms resolution.

``build_chip`` is the generator of chip instances: random sparse input
synapses, per-neuron device mismatch, and one global synaptic gain calibrated
so a target fraction of (neuron, state) pairs is driven above threshold under
random behaviour. Chips differ only through the seed. Learning methods never
receive a chip's arrays; they receive its filtered spike trains.

Chip keys (N hidden units, K cells per input dimension)::

    enc_centers [6, K]  enc_sigma [6, K]  enc_max_rate_hz [6, K]   w_in [N, 6K]
    bias, tau_mem_ms, tau_syn_ms, tau_adapt_ms, adapt_increment,
    threshold, refractory_ms, tau_trace_ms                          [N] each
"""
from __future__ import annotations
import numpy as np
from data.air_hockey_env import CONDITIONS, AirHockeySlots, sample_initial_conditions

STATE_DIM = 6
SUBSTEPS = 20
CELLS_PER_DIM = 10
INPUT_RATE_HZ = 200.
CONNECTION_PROB = .2
INHIBITORY_FRACTION = .2
TARGET_ACTIVE_FRACTION = .25
TAU_MEM_MS = 20.
TAU_SYN_MS = 5.
TAU_ADAPT_MS = 100.
ADAPT_INCREMENT = .002
REFRACTORY_MS = 2.
TAU_TRACE_MS = 30.
MISMATCH_CV = .2
CALIBRATION_EPISODES = 16
PER_NEURON = ("bias", "tau_mem_ms", "tau_syn_ms", "tau_adapt_ms", "adapt_increment",
              "threshold", "refractory_ms", "tau_trace_ms")


def encoder_ranges(condition):
    """Population-code range per observed variable.

    The lateral-velocity code spans the condition's speed range in both
    directions (the puck rebounds from the side walls), so wider speed ranges
    are represented at coarser resolution by the same 10 cells (Sec. III-B).
    """
    vmax = max(0.2, float(CONDITIONS[condition]["speed"][1]))
    return np.array([[.80, 1.60], [-.50, .50], [-1., 4.], [-vmax, vmax], [.70, 1.50], [-.05, .05]])


def random_behaviour_states(condition, rng, episodes):
    env = AirHockeySlots(episodes)
    env.reset(np.arange(episodes), sample_initial_conditions(condition, rng, episodes))
    states = [env.obs.copy()]
    while not np.all(env.done):
        rows = np.flatnonzero(~env.done)
        env.step(rows, rng.integers(0, 2, size=len(rows)))
        states.append(env.obs[~env.done].copy())
    return np.concatenate(states, axis=0)


def build_chip(condition, seed, n_hidden=1020):
    """Sample one fixed random substrate ("chip instance") for a condition."""
    rng = np.random.default_rng(seed)
    k, n = CELLS_PER_DIM, int(n_hidden)
    ranges = encoder_ranges(condition)
    centers = np.stack([np.linspace(lo, hi, k) for lo, hi in ranges])
    spacing = (ranges[:, 1] - ranges[:, 0]) / (k - 1)
    sigma = np.repeat(spacing[:, None], k, axis=1)
    rates = np.full((STATE_DIM, k), INPUT_RATE_HZ)
    n_in = STATE_DIM * k
    mask = rng.random((n, n_in)) < CONNECTION_PROB
    sign = np.where(rng.random((n, n_in)) < INHIBITORY_FRACTION, -1., 1.)
    w_shape = mask * sign * rng.lognormal(0., .25, (n, n_in))
    tau_mem = TAU_MEM_MS * rng.lognormal(0., MISMATCH_CV, n)
    tau_syn = TAU_SYN_MS * rng.lognormal(0., MISMATCH_CV, n)
    tau_adapt = TAU_ADAPT_MS * rng.lognormal(0., MISMATCH_CV, n)
    threshold = rng.lognormal(0., .5 * MISMATCH_CV, n)
    adapt_inc = ADAPT_INCREMENT * rng.lognormal(0., MISMATCH_CV, n)
    # Global gain calibration over states visited by random behaviour.
    states = random_behaviour_states(condition, rng, CALIBRATION_EPISODES)
    if states.shape[0] > 600:
        states = states[rng.choice(states.shape[0], 600, replace=False)]
    dim_index = np.repeat(np.arange(STATE_DIM), k)
    z = (states[:, dim_index] - centers.reshape(-1)) / sigma.reshape(-1)
    in_rate = rates.reshape(-1) * 1e-3 * np.exp(-.5 * z * z)
    a_mem, a_syn = np.exp(-1. / tau_mem), np.exp(-1. / tau_syn)
    drive = (in_rate @ w_shape.T) / ((1. - a_syn) * (1. - a_mem))
    ratio = np.where(drive > 0, threshold / np.maximum(drive, 1e-12), np.inf)
    gain = float(np.quantile(ratio, TARGET_ACTIVE_FRACTION))
    if not np.isfinite(gain):
        gain = float(np.quantile(ratio[np.isfinite(ratio)], .5))
    chip = {
        "enc_centers": centers, "enc_sigma": sigma, "enc_max_rate_hz": rates,
        "w_in": gain * w_shape, "bias": np.zeros(n), "tau_mem_ms": tau_mem,
        "tau_syn_ms": tau_syn, "tau_adapt_ms": tau_adapt, "adapt_increment": adapt_inc,
        "threshold": threshold, "refractory_ms": np.full(n, REFRACTORY_MS),
        "tau_trace_ms": np.full(n, TAU_TRACE_MS),
    }
    return {key: np.asarray(value, dtype=np.float32) for key, value in chip.items()}


def _decay(tau_ms):
    return np.exp(-1. / np.asarray(tau_ms, dtype=np.float64)).astype(np.float32)


class ChipBank:
    """Run ``C`` chips (one episode slot each) in lockstep at 1 ms resolution."""

    def __init__(self, chips):
        n = chips[0]["w_in"].shape[0]
        k = chips[0]["enc_centers"].shape[1]
        for chip in chips:
            if chip["w_in"].shape != (n, STATE_DIM * k):
                raise ValueError("all chips in a bank must have the same shape")
        self.C, self.n = len(chips), n

        def stack(key):
            return np.stack([np.asarray(c[key], dtype=np.float32) for c in chips])

        self.dim_index = np.repeat(np.arange(STATE_DIM), k)
        self.centers = stack("enc_centers").reshape(self.C, -1)
        self.inv_sigma = (1. / stack("enc_sigma")).reshape(self.C, -1).astype(np.float32)
        self.rate_per_ms = (stack("enc_max_rate_hz") * 1e-3).reshape(self.C, -1).astype(np.float32)
        self.w_in_t = np.ascontiguousarray(np.transpose(stack("w_in"), (0, 2, 1)))
        self.a_mem = _decay(stack("tau_mem_ms"))
        self.a_syn = _decay(stack("tau_syn_ms"))
        self.a_adapt = _decay(stack("tau_adapt_ms"))
        self.a_trace = _decay(stack("tau_trace_ms"))
        self.bias = stack("bias")
        self.threshold = stack("threshold")
        self.adapt_inc = stack("adapt_increment")
        self.ref_steps = np.rint(stack("refractory_ms")).astype(np.float32)
        shape_in, shape = (self.C, self.centers.shape[1]), (self.C, n)
        self.phase = np.zeros(shape_in, dtype=np.float32)
        self.i_syn = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.adapt = np.zeros(shape, dtype=np.float32)
        self.refractory = np.zeros(shape, dtype=np.float32)
        self.trace = np.zeros(shape, dtype=np.float32)
        self._spk = np.zeros(shape, dtype=np.float32)
        self._tmp = np.zeros(shape, dtype=np.float32)
        self._mask = np.zeros(shape, dtype=bool)

    def reset(self, rows):
        rows = np.atleast_1d(np.asarray(rows, dtype=int))
        for arr in (self.phase, self.i_syn, self.v, self.adapt, self.refractory, self.trace):
            arr[rows] = 0.

    def step(self, obs):
        """Run 20 ms driven by ``obs`` [C, 6]; return filtered spike trains [C, N]."""
        x = np.asarray(obs, dtype=np.float32)[:, self.dim_index]
        z = (x - self.centers) * self.inv_sigma
        rate = self.rate_per_ms * np.exp(-.5 * z * z)
        # Regular input spike trains: an accumulating phase emits a spike at 1.
        s_in = np.empty((SUBSTEPS,) + rate.shape, dtype=np.float32)
        phase = self.phase
        for t in range(SUBSTEPS):
            phase += rate
            np.greater_equal(phase, 1., out=s_in[t], casting="unsafe")
            phase -= s_in[t]
        drive = np.matmul(np.ascontiguousarray(s_in.transpose(1, 0, 2)), self.w_in_t)  # [C, 20, N]
        i_syn, v, adapt, refr, trace = self.i_syn, self.v, self.adapt, self.refractory, self.trace
        spk, tmp, mask = self._spk, self._tmp, self._mask
        for t in range(SUBSTEPS):
            i_syn *= self.a_syn
            i_syn += drive[:, t]
            v *= self.a_mem
            v += i_syn
            v += self.bias
            v -= adapt
            np.copyto(v, 0., where=refr > 0.)
            np.greater_equal(v, self.threshold, out=mask)
            np.copyto(spk, mask, casting="unsafe")
            np.copyto(v, 0., where=mask)
            adapt *= self.a_adapt
            np.multiply(self.adapt_inc, spk, out=tmp)
            adapt += tmp
            refr -= 1.
            np.maximum(refr, 0., out=refr)
            np.copyto(refr, self.ref_steps, where=mask)
            trace *= self.a_trace
            trace += spk
        return trace.copy()
