"""Software reproduction of the paper's spiking e-prop RL controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.air_hockey_env import state_ranges


@dataclass
class SpikingAgentConfig:
    state_dim: int = 6
    cells_per_dim: int = 10
    hidden_neurons: int = 1020
    actions: int = 2
    sim_dt_ms: float = 1.0
    control_ms: int = 20
    tau_mem_ms: float = 10.0
    tau_trace_ms: float = 30.0
    input_rate_hz: float = 250.0
    encoding_sigma_cells: float = 1.5
    input_weight_scale: float = 0.55
    recurrent_weight_scale: float = 0.04
    learning_rate: float = 0.0015
    gamma: float = 0.96
    update_every_trials: int = 2
    entropy_floor: float = 0.04
    center_returns: bool = True


class SpikingEpropAgent:
    """Fixed random LIF reservoir plus plastic softmax readout.

    The hidden layer is a software stand-in for the DYNAP-SE silicon neurons:
    input population spike trains drive leaky integrate-and-fire dynamics with
    fixed random connectivity.  Only hidden-to-readout weights are updated using
    the e-prop-style policy-gradient rule given in Eq. 1 of the paper.
    """

    def __init__(self, seed: int = 0, config: SpikingAgentConfig | None = None) -> None:
        self.config = config or SpikingAgentConfig()
        self.rng = np.random.default_rng(seed)
        cfg = self.config
        self.input_neurons = cfg.state_dim * cfg.cells_per_dim
        self.low, self.high = state_ranges()
        self.centers = np.stack(
            [np.linspace(self.low[i], self.high[i], cfg.cells_per_dim) for i in range(cfg.state_dim)]
        )
        fan_in = max(1, self.input_neurons)
        self.w_in = self.rng.normal(0.0, cfg.input_weight_scale / np.sqrt(fan_in), (cfg.hidden_neurons, self.input_neurons))
        # Store sparse random recurrence as edge lists.  This preserves fixed
        # random reservoir dynamics without doing a dense 1020x1020 multiply at
        # every simulated millisecond.
        rec_edges = cfg.hidden_neurons * 16
        self.rec_src = self.rng.integers(0, cfg.hidden_neurons, rec_edges)
        self.rec_dst = self.rng.integers(0, cfg.hidden_neurons, rec_edges)
        self.rec_w = self.rng.normal(0.0, cfg.recurrent_weight_scale, rec_edges)
        self.w_out = self.rng.normal(0.0, 0.01, (cfg.hidden_neurons, cfg.actions))
        self.v = self.rng.uniform(0.0, 0.5, cfg.hidden_neurons)
        self.trace = np.zeros(cfg.hidden_neurons, dtype=np.float64)
        self.prev_spikes = np.zeros(cfg.hidden_neurons, dtype=np.float64)
        self.pending_episodes: list[list[dict[str, np.ndarray | int | float]]] = []

    def encode_rates(self, state: np.ndarray) -> np.ndarray:
        cfg = self.config
        clipped = np.clip(state[: cfg.state_dim], self.low, self.high)
        rates = []
        for dim in range(cfg.state_dim):
            dist = (self.centers[dim] - clipped[dim]) / (cfg.encoding_sigma_cells * (self.high[dim] - self.low[dim]) / (cfg.cells_per_dim - 1))
            activity = np.exp(-0.5 * dist * dist)
            rates.append(activity)
        return cfg.input_rate_hz * np.concatenate(rates)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / np.sum(exp)
        eps = self.config.entropy_floor
        return (1.0 - eps) * probs + eps / self.config.actions

    def act(self, state: np.ndarray, greedy: bool = False) -> tuple[int, np.ndarray, np.ndarray]:
        cfg = self.config
        rates = self.encode_rates(state)
        # Aggregate the deterministic FPGA spike replay over one 20 ms control
        # window.  This is the same state update granularity at which the paper
        # reads out actions, but avoids a costly Python loop over every hardware
        # millisecond.
        expected_input_spikes = rates * (cfg.control_ms / 1000.0)
        input_spikes = self.rng.poisson(expected_input_spikes).astype(np.float64)
        rec_current = np.zeros(cfg.hidden_neurons, dtype=np.float64)
        active_edges = self.prev_spikes[self.rec_src] > 0.0
        if np.any(active_edges):
            rec_current += np.bincount(
                self.rec_dst[active_edges],
                weights=self.rec_w[active_edges],
                minlength=cfg.hidden_neurons,
            )
        alpha_mem = np.exp(-cfg.control_ms / cfg.tau_mem_ms)
        alpha_trace = np.exp(-cfg.control_ms / cfg.tau_trace_ms)
        current = self.w_in @ input_spikes + rec_current
        self.v = alpha_mem * self.v + current
        spike_prob = 1.0 / (1.0 + np.exp(-(self.v - 1.0) * 2.5))
        spikes = self.rng.random(cfg.hidden_neurons) < spike_prob
        self.v[spikes] = 0.0
        self.prev_spikes = spikes.astype(np.float64)
        self.trace = alpha_trace * self.trace + self.prev_spikes

        filtered = self.trace
        logits = filtered @ self.w_out
        probs = self._softmax(logits)
        action = int(np.argmax(probs)) if greedy else int(self.rng.choice(cfg.actions, p=probs))
        return action, probs.copy(), filtered.copy()

    def finish_episode(self, trajectory: list[dict[str, np.ndarray | int | float]]) -> None:
        self.pending_episodes.append(trajectory)
        if len(self.pending_episodes) >= self.config.update_every_trials:
            self.update_readout(self.pending_episodes)
            self.pending_episodes = []

    def update_readout(self, episodes: list[list[dict[str, np.ndarray | int | float]]]) -> None:
        cfg = self.config
        grad = np.zeros_like(self.w_out)
        for episode in episodes:
            returns = []
            g = 0.0
            for item in reversed(episode):
                g = float(item["reward"]) + cfg.gamma * g
                returns.append(g)
            returns.reverse()
            if cfg.center_returns and len(returns) > 1:
                returns_array = np.asarray(returns, dtype=np.float64)
                returns_array = (returns_array - returns_array.mean()) / (returns_array.std() + 1e-6)
                returns = returns_array.tolist()
            for item, ret in zip(episode, returns):
                probs = np.asarray(item["probs"], dtype=np.float64)
                chosen = int(item["action"])
                error = probs.copy()
                error[chosen] -= 1.0
                grad += ret * np.outer(np.asarray(item["trace"], dtype=np.float64), error)
        scale = max(1, sum(len(ep) for ep in episodes))
        self.w_out -= cfg.learning_rate * grad / scale

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            w_out=self.w_out,
            v=self.v,
            trace=self.trace,
            prev_spikes=self.prev_spikes,
        )
