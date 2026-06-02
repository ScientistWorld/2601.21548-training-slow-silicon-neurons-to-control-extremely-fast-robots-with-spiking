"""Lightweight air-hockey interception task used by the reproduction.

The paper evaluates a robot air-hockey task in MuJoCo.  The exact robot
simulator and DYNAP-SE hardware are not available in this benchmark, so this
module provides a deterministic task-level simulator with the same observable
state, two motion primitives, reward, and success condition.  It is intentionally
method-agnostic and is used by both baselines and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AirHockeyConfig:
    control_dt: float = 0.02
    episode_steps: int = 100
    puck_radius: float = 0.04
    mallet_radius: float = 0.04
    table_goal_x: float = 1.5
    home_x: float = 0.70
    forward_x: float = 1.50
    center_y: float = 0.0
    mallet_speed: float = 4.0
    active_hit_speed: float = 0.35
    swing_active_start: int = 6
    swing_active_end: int = 11
    rebound_speed: float = 1.25
    terminal_reward: float = 20.0
    step_penalty: float = -0.1
    shaping_scale: float = 0.2


CONDITIONS = {
    "stationary_1m": {
        "puck_x": (1.00, 1.00),
        "puck_y": (0.00, 0.00),
        "speed": (0.0, 0.0),
    },
    "constant_speed": {
        "puck_x": (1.95, 1.95),
        "puck_y": (0.00, 0.00),
        "speed": (1.20, 1.20),
    },
    "variable_speed": {
        "puck_x": (1.95, 1.95),
        "puck_y": (0.00, 0.00),
        "speed": (1.0, 1.5),
    },
    "random_position_speed": {
        "puck_x": (1.90, 2.00),
        "puck_y": (-0.05, 0.05),
        "speed": (1.0, 1.5),
    },
    "speed_narrow": {
        "puck_x": (1.95, 1.95),
        "puck_y": (0.00, 0.00),
        "speed": (0.7, 0.9),
    },
    "speed_medium": {
        "puck_x": (1.95, 1.95),
        "puck_y": (0.00, 0.00),
        "speed": (0.7, 1.2),
    },
    "speed_wide": {
        "puck_x": (1.95, 1.95),
        "puck_y": (0.00, 0.00),
        "speed": (0.7, 1.5),
    },
}


class AirHockeyInterceptionEnv:
    """Task-level simulator for the paper's 6D state and two primitives."""

    def __init__(
        self,
        condition: str = "random_position_speed",
        seed: int = 0,
        config: AirHockeyConfig | None = None,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}; choose from {sorted(CONDITIONS)}")
        self.condition = condition
        self.config = config or AirHockeyConfig()
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(6, dtype=np.float32)
        self.steps = 0
        self.success = False
        self.hit = False
        self.primitive = 0
        self.swing_age = -1

    @property
    def hit_radius(self) -> float:
        return self.config.puck_radius + self.config.mallet_radius

    def reset(self) -> np.ndarray:
        spec = CONDITIONS[self.condition]
        puck_x = self.rng.uniform(*spec["puck_x"])
        puck_y = self.rng.uniform(*spec["puck_y"])
        speed = self.rng.uniform(*spec["speed"])
        # Puck starts on the far side and travels toward the robot.  A hit turns
        # vx positive, matching the paper's success reward definition.
        vx = -abs(speed)
        if speed == 0.0:
            vx = 0.0
        self.state[:] = [puck_x, puck_y, vx, 0.0, self.config.home_x, self.config.center_y]
        self.steps = 0
        self.success = False
        self.hit = False
        self.primitive = 0
        self.swing_age = -1
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float | bool]]:
        cfg = self.config
        px, py, vx, vy, ex, ey = [float(v) for v in self.state]
        action = int(action)
        if action == 0:
            self.primitive = 0
            self.swing_age = -1
        elif self.primitive == 0 and ex <= cfg.home_x + 0.06:
            self.primitive = 1
            self.swing_age = 0
        elif self.primitive == 1:
            self.swing_age += 1

        target_x = cfg.home_x if self.primitive == 0 else cfg.forward_x
        target_y = cfg.center_y

        max_delta = cfg.mallet_speed * cfg.control_dt
        old_ex = ex
        old_ey = ey
        delta = np.array([target_x - ex, target_y - ey], dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        if norm > max_delta and norm > 0.0:
            delta *= max_delta / norm
        ex += float(delta[0])
        ey += float(delta[1])
        mallet_vx = (ex - old_ex) / cfg.control_dt
        mallet_vy = (ey - old_ey) / cfg.control_dt

        px += vx * cfg.control_dt
        py += vy * cfg.control_dt
        dist = float(np.hypot(px - ex, py - ey))

        reward = cfg.step_penalty
        swing_in_contact_phase = cfg.swing_active_start <= self.swing_age <= cfg.swing_active_end
        active_forward_swing = mallet_vx > cfg.active_hit_speed and swing_in_contact_phase
        active_lateral_alignment = abs(mallet_vy) < 0.5
        if (not self.hit) and dist <= self.hit_radius and active_forward_swing and active_lateral_alignment:
            self.hit = True
            vx = abs(vx) * cfg.rebound_speed + 0.15
            reward = cfg.shaping_scale * vx

        px += vx * cfg.control_dt
        py += vy * cfg.control_dt
        if px > cfg.table_goal_x and vx > 0.1:
            self.success = True
            reward = cfg.terminal_reward

        self.steps += 1
        done = self.success or self.steps >= cfg.episode_steps or px < 0.0 or abs(py) > 0.4
        self.state[:] = [px, py, vx, vy, ex, ey]
        info = {"success": self.success, "hit": self.hit, "steps": self.steps, "mallet_vx": mallet_vx}
        return self.state.copy(), float(reward), bool(done), info


def state_ranges() -> tuple[np.ndarray, np.ndarray]:
    """Encoding ranges used for population coding."""

    low = np.array([0.0, -0.25, -1.6, -0.5, 0.55, -0.15], dtype=np.float32)
    high = np.array([2.05, 0.25, 2.2, 0.5, 1.55, 0.15], dtype=np.float32)
    return low, high
