"""FROZEN COPY for held-out scoring (identical to data/air_hockey_env.py at
freeze time; edits to data/ do not affect the test).

Task-level air-hockey interception simulator (public training environment).

The paper (Sec. II-A, II-C, Fig. 1, Table I) trains in a MuJoCo air-hockey
environment with a robot arm whose end effector executes open-loop spline
primitives.  The robot and the DYNAP-SE hardware are not available here, so
this module reproduces the task at the level the controller interacts with:

* robot frame at the near end of a 1.948 m x 1.038 m table; x points forward
  along the table, y is lateral;
* puck radius 0.03165 m, mallet radius 0.04815 m (contact distance 0.0798 m);
* the puck is generated in the band x in [0.95, 1.05] m at the table edge
  and launched *laterally* across the table (the "pucks' generation area" of
  Fig. 1); it moves without friction and bounces elastically off the walls;
* two motion primitives: action 0 drives the end effector to the home point
  (0.70, 0), action 1 to the forward point (1.50, 0).  A primitive is an
  open-loop cubic spline in task space from the current end-effector position
  and velocity to the target, ending at rest; a new spline starts only when
  the selected target changes, so corrective maneuvers are compositions of
  primitives;
* 50 Hz decisions, 1 ms physics integration, elastic puck-mallet collisions;
* reward of Eq. 2 and success ``x_p > 1.5 m and v_x > 0.1 m/s``; an episode
  lasts at most 100 control steps (2 s).

Everything is vectorised over a batch of independent episodes, so the same
code serves single-episode online learning and batched evaluation.  The module
is method-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AirHockeyConfig:
    table_length: float = 1.948
    table_width: float = 1.038
    puck_radius: float = 0.03165
    mallet_radius: float = 0.04815
    home_x: float = 0.70
    forward_x: float = 1.50
    launch_y: float = 0.45
    goal_x: float = 1.5
    goal_vx: float = 0.1
    control_dt: float = 0.02
    physics_substeps: int = 20
    mallet_mean_speed: float = 2.0
    min_primitive_s: float = 0.10
    max_primitive_s: float = 0.60
    restitution: float = 1.0
    max_steps: int = 100
    terminal_reward: float = 20.0
    contact_reward_scale: float = 0.2
    step_penalty: float = -0.1


# Initial-condition distributions of the paper's experimental conditions
# (Sec. III).  ``puck_x`` is the launch position along the table inside the
# generation band, ``speed`` the lateral launch speed in m/s.
CONDITIONS = {
    "stationary_1m": {"puck_x": (1.00, 1.00), "speed": (0.0, 0.0)},
    "constant_speed": {"puck_x": (1.00, 1.00), "speed": (1.2, 1.2)},
    "variable_speed": {"puck_x": (1.00, 1.00), "speed": (1.0, 1.5)},
    "random_position_speed": {"puck_x": (0.95, 1.05), "speed": (1.0, 1.5)},
    "speed_narrow": {"puck_x": (1.00, 1.00), "speed": (0.7, 0.9)},
    "speed_medium": {"puck_x": (1.00, 1.00), "speed": (0.7, 1.2)},
    "speed_wide": {"puck_x": (1.00, 1.00), "speed": (0.7, 1.5)},
    "speed_extreme": {"puck_x": (1.00, 1.00), "speed": (0.5, 2.0)},
}


def sample_initial_conditions(condition: str, rng: np.random.Generator, n: int) -> np.ndarray:
    """Return an ``(n, 2)`` array of (launch x, launch speed)."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {sorted(CONDITIONS)}")
    spec = CONDITIONS[condition]
    x0 = rng.uniform(*spec["puck_x"], size=n)
    speed = rng.uniform(*spec["speed"], size=n)
    return np.stack([x0, speed], axis=1)


class AirHockeyBatch:
    """A batch of independent interception episodes advanced in lock-step."""

    def __init__(self, init: np.ndarray, config: AirHockeyConfig | None = None) -> None:
        cfg = config or AirHockeyConfig()
        self.cfg = cfg
        init = np.asarray(init, dtype=np.float64).reshape(-1, 2)
        b = init.shape[0]
        self.batch = b
        self.px = init[:, 0].copy()
        self.vy = -np.abs(init[:, 1])
        self.py = np.where(init[:, 1] == 0.0, 0.0, cfg.launch_y)
        self.vx = np.zeros(b)
        self.mx = np.full(b, cfg.home_x)
        self.mvx = np.zeros(b)
        self.target = np.zeros(b, dtype=np.int64)
        self.sp_t0 = np.zeros(b)
        self.sp_p0 = np.full(b, cfg.home_x)
        self.sp_v0 = np.zeros(b)
        self.sp_p1 = np.full(b, cfg.home_x)
        self.sp_T = np.ones(b)
        self.time = np.zeros(b)
        self.steps = np.zeros(b, dtype=np.int64)
        self.hit = np.zeros(b, dtype=bool)
        self.success = np.zeros(b, dtype=bool)
        self.done = np.zeros(b, dtype=bool)
        self._targets = np.array([cfg.home_x, cfg.forward_x])
        self._dt = cfg.control_dt / cfg.physics_substeps
        self._contact = cfg.puck_radius + cfg.mallet_radius
        self._wall_y = 0.5 * cfg.table_width - cfg.puck_radius
        self._wall_x_lo = cfg.puck_radius
        self._wall_x_hi = cfg.table_length - cfg.puck_radius

    def observe(self) -> np.ndarray:
        """Observed 6D state ``[x_p, y_p, v_x, v_y, x_ee, y_ee]`` per episode."""
        return np.stack([self.px, self.py, self.vx, self.vy, self.mx, np.zeros(self.batch)], axis=1)

    def _mallet(self, idx: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = self.sp_T[idx]
        s = np.clip((t - self.sp_t0[idx]) / T, 0.0, 1.0)
        p0, v0, p1 = self.sp_p0[idx], self.sp_v0[idx], self.sp_p1[idx]
        s2 = s * s
        s3 = s2 * s
        pos = (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * T * v0 + (-2 * s3 + 3 * s2) * p1
        vel = ((6 * s2 - 6 * s) * p0 + (3 * s2 - 4 * s + 1) * T * v0 + (-6 * s2 + 6 * s) * p1) / T
        finished = s >= 1.0
        pos = np.where(finished, p1, pos)
        vel = np.where(finished, 0.0, vel)
        return pos, vel

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Apply one 20 ms control step to every unfinished episode.

        Returns ``(obs, reward, done, info)``; finished episodes are frozen and
        receive zero reward.
        """
        cfg = self.cfg
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.shape[0] != self.batch:
            raise ValueError("one action per episode is required")
        if np.any((actions < 0) | (actions > 1)):
            raise ValueError("actions must be 0 (home) or 1 (forward)")
        live = ~self.done
        idx = np.flatnonzero(live)
        reward = np.zeros(self.batch)
        if idx.size == 0:
            return self.observe(), reward, self.done.copy(), self._info()

        # Start a new spline where the selected primitive changed.
        changed = idx[actions[idx] != self.target[idx]]
        if changed.size:
            p1 = self._targets[actions[changed]]
            dist = np.abs(p1 - self.mx[changed])
            self.sp_T[changed] = np.clip(dist / cfg.mallet_mean_speed, cfg.min_primitive_s, cfg.max_primitive_s)
            self.sp_t0[changed] = self.time[changed]
            self.sp_p0[changed] = self.mx[changed]
            self.sp_v0[changed] = self.mvx[changed]
            self.sp_p1[changed] = p1
            self.target[changed] = actions[changed]

        min_dist = np.full(idx.size, np.inf)
        moving = np.ones(idx.size, dtype=bool)
        dt = self._dt
        e1 = 1.0 + cfg.restitution
        for _ in range(cfg.physics_substeps):
            sub = idx[moving]
            if sub.size == 0:
                break
            t = self.time[sub] + dt
            self.time[sub] = t
            mx, mvx = self._mallet(sub, t)
            self.mx[sub] = mx
            self.mvx[sub] = mvx
            px = self.px[sub] + self.vx[sub] * dt
            py = self.py[sub] + self.vy[sub] * dt
            vx = self.vx[sub]
            vy = self.vy[sub]
            # Elastic, frictionless table walls.
            over = py > self._wall_y
            py = np.where(over, 2 * self._wall_y - py, py)
            vy = np.where(over, -np.abs(vy), vy)
            under = py < -self._wall_y
            py = np.where(under, -2 * self._wall_y - py, py)
            vy = np.where(under, np.abs(vy), vy)
            low = px < self._wall_x_lo
            px = np.where(low, 2 * self._wall_x_lo - px, px)
            vx = np.where(low, np.abs(vx), vx)
            high = px > self._wall_x_hi
            px = np.where(high, 2 * self._wall_x_hi - px, px)
            vx = np.where(high, -np.abs(vx), vx)
            # Puck-mallet contact (mallet driven kinematically, infinite mass).
            dx = px - mx
            dy = py
            dist = np.hypot(dx, dy)
            pos_in_sub = np.flatnonzero(moving)
            min_dist[pos_in_sub] = np.minimum(min_dist[pos_in_sub], dist)
            safe = np.where(dist > 0.0, dist, 1.0)
            nx = dx / safe
            ny = dy / safe
            rel = (vx - mvx) * nx + vy * ny
            contact = (dist <= self._contact) & (dist > 0.0) & (rel < 0.0)
            if np.any(contact):
                vx = np.where(contact, vx - e1 * rel * nx, vx)
                vy = np.where(contact, vy - e1 * rel * ny, vy)
                px = np.where(contact, mx + nx * self._contact, px)
                py = np.where(contact, ny * self._contact, py)
                self.hit[sub[contact]] = True
            self.px[sub] = px
            self.py[sub] = py
            self.vx[sub] = vx
            self.vy[sub] = vy
            scored = (px > cfg.goal_x) & (vx > cfg.goal_vx)
            if np.any(scored):
                self.success[sub[scored]] = True
                moving[pos_in_sub[scored]] = False

        self.steps[idx] += 1
        succ = self.success[idx]
        near = min_dist <= self._contact
        r = np.where(near, cfg.contact_reward_scale * self.vx[idx], cfg.step_penalty)
        r = np.where(succ, cfg.terminal_reward, r)
        reward[idx] = r
        self.done[idx] = succ | (self.steps[idx] >= cfg.max_steps)
        return self.observe(), reward, self.done.copy(), self._info()

    def _info(self) -> dict[str, np.ndarray]:
        return {"success": self.success.copy(), "hit": self.hit.copy(), "steps": self.steps.copy()}


class AirHockeyInterceptionEnv:
    """Single-episode convenience wrapper with a gym-like interface."""

    def __init__(self, condition: str = "random_position_speed", seed: int = 0,
                 config: AirHockeyConfig | None = None) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}; choose from {sorted(CONDITIONS)}")
        self.condition = condition
        self.config = config or AirHockeyConfig()
        self.rng = np.random.default_rng(seed)
        self._batch: AirHockeyBatch | None = None

    def reset(self) -> np.ndarray:
        init = sample_initial_conditions(self.condition, self.rng, 1)
        self._batch = AirHockeyBatch(init, self.config)
        return self._batch.observe()[0]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert self._batch is not None, "call reset() first"
        obs, reward, done, info = self._batch.step(np.array([int(action)]))
        return obs[0], float(reward[0]), bool(done[0]), {k: v[0].item() for k, v in info.items()}
