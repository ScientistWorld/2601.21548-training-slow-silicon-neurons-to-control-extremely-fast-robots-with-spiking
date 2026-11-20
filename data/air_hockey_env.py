"""Public air-hockey interception task: torque-limited planar robot in MuJoCo.

The six observations, two task-space cubic primitives, 50 Hz decisions,
1 kHz physics, reward (Eq. 2) and success event follow the paper (Sec. II-A,
II-C, Fig. 1, Table I). The robot geometry is a documented software substitute;
see robot/README.md. No test chips, test episodes or test seeds are stored here.

``AirHockeySlots`` holds a fixed number of independent episode slots that are
stepped together; each slot can be reset on its own, so many sequential
learning runs (one per controller instance) advance in lockstep.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
import mujoco
from mujoco import rollout


@dataclass(frozen=True)
class AirHockeyConfig:
    table_length: float = 1.948
    table_width: float = 1.038
    puck_radius: float = .03165
    mallet_radius: float = .04815
    home_x: float = .70
    forward_x: float = 1.50
    launch_y: float = .45
    goal_x: float = 1.5
    goal_vx: float = .1
    control_dt: float = .02
    physics_substeps: int = 20
    mallet_mean_speed: float = 2.
    min_primitive_s: float = .10
    max_primitive_s: float = .60
    max_steps: int = 100
    terminal_reward: float = 20.
    contact_reward_scale: float = .2
    step_penalty: float = -.1


# Launch conditions of the paper (Sec. III). The puck starts at (puck_x, 0.45 m)
# and moves laterally toward the robot's centre line at the sampled speed.
CONDITIONS = {
    "stationary_1m": {"puck_x": (1., 1.), "speed": (0., 0.)},
    "constant_speed": {"puck_x": (1., 1.), "speed": (1.2, 1.2)},
    "variable_speed": {"puck_x": (1., 1.), "speed": (1., 1.5)},
    "random_position_speed": {"puck_x": (.95, 1.05), "speed": (1., 1.5)},
    "speed_narrow": {"puck_x": (1., 1.), "speed": (.7, .9)},
    "speed_medium": {"puck_x": (1., 1.), "speed": (.7, 1.2)},
    "speed_wide": {"puck_x": (1., 1.), "speed": (.7, 1.5)},
    "speed_extreme": {"puck_x": (1., 1.), "speed": (.5, 2.)},
}


def sample_initial_conditions(condition, rng, n):
    """Return ``[n, 2]`` launches: puck x position (m) and lateral speed (m/s)."""
    spec = CONDITIONS[condition]
    return np.column_stack((rng.uniform(*spec["puck_x"], n), rng.uniform(*spec["speed"], n)))


def inverse_kinematics(x, y):
    wx = np.asarray(x) - .4
    y = np.asarray(y)
    c = np.clip((wx * wx + y * y - .75 ** 2 - .55 ** 2) / (2 * .75 * .55), -.99999, .99999)
    q2 = np.arccos(c)
    q1 = np.arctan2(y, wx) - np.arctan2(.55 * np.sin(q2), .75 + .55 * c)
    return np.clip(np.stack((q1, q2, -q1 - q2), axis=-1), -3.05, 3.05)


@lru_cache(maxsize=1)
def _physics():
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / 'robot' / 'air_hockey.xml'))
    return model, mujoco.MjData(model), rollout.Rollout(nthread=0)


class AirHockeySlots:
    """``n`` independent episodes; ``reset`` and ``step`` act on chosen slots."""

    def __init__(self, n, config=None):
        cfg = self.cfg = config or AirHockeyConfig()
        self.model, self.data, self.engine = _physics()
        n = self.n = int(n)
        # FULLPHYSICS state: time, qpos (puck x, y, three joints), qvel.
        self.state = np.zeros((n, 1 + self.model.nq + self.model.nv))
        self.time = np.zeros(n)
        self.steps = np.zeros(n, dtype=int)
        self.done = np.ones(n, dtype=bool)
        self.success = np.zeros(n, dtype=bool)
        self.hit = np.zeros(n, dtype=bool)
        self.target = np.zeros(n, dtype=int)
        self.sp_t0 = np.zeros(n)
        self.sp_p0 = np.full(n, cfg.home_x)
        self.sp_v0 = np.zeros(n)
        self.sp_p1 = np.full(n, cfg.home_x)
        self.sp_T = np.ones(n)
        self.mx = np.full(n, cfg.home_x)
        self.my = np.zeros(n)
        self.mvx = np.zeros(n)
        self.obs = np.zeros((n, 6))

    def reset(self, rows, init):
        cfg = self.cfg
        rows = np.atleast_1d(np.asarray(rows, dtype=int))
        init = np.asarray(init, dtype=np.float64).reshape(len(rows), 2)
        m = len(rows)
        st = np.zeros((m, self.state.shape[1]))
        st[:, 1] = init[:, 0]
        st[:, 2] = np.where(init[:, 1] == 0, 0., cfg.launch_y)
        st[:, 3:6] = inverse_kinematics(np.full(m, cfg.home_x), np.zeros(m))
        st[:, 7] = -init[:, 1]
        self.state[rows] = st
        self.time[rows] = 0.
        self.steps[rows] = 0
        self.done[rows] = False
        self.success[rows] = False
        self.hit[rows] = False
        self.target[rows] = 0
        self.sp_t0[rows] = 0.
        self.sp_p0[rows] = cfg.home_x
        self.sp_v0[rows] = 0.
        self.sp_p1[rows] = cfg.home_x
        self.sp_T[rows] = 1.
        self.mx[rows] = cfg.home_x
        self.my[rows] = 0.
        self.mvx[rows] = 0.
        self.obs[rows] = np.column_stack((st[:, 1:3], st[:, 6:8], self.mx[rows], self.my[rows]))

    def step(self, rows, actions):
        """Advance the given running slots by one 20 ms control tick.

        Returns the Eq. 2 reward and the episode-finished flag of each row.
        """
        cfg = self.cfg
        idx = np.atleast_1d(np.asarray(rows, dtype=int))
        actions = np.asarray(actions, dtype=int).reshape(-1)
        if len(actions) != len(idx) or np.any((actions < 0) | (actions > 1)):
            raise ValueError('one binary action per stepped slot is required')
        if np.any(self.done[idx]):
            raise ValueError('cannot step a finished episode')
        if not len(idx):
            return np.zeros(0), np.zeros(0, dtype=bool)
        sel = actions != self.target[idx]
        changed = idx[sel]
        self.sp_p0[changed] = self.mx[changed]
        self.sp_v0[changed] = self.mvx[changed]
        self.sp_p1[changed] = np.where(actions[sel], cfg.forward_x, cfg.home_x)
        self.sp_T[changed] = np.clip(abs(self.sp_p1[changed] - self.mx[changed]) / cfg.mallet_mean_speed,
                                     cfg.min_primitive_s, cfg.max_primitive_s)
        self.sp_t0[changed] = self.time[changed]
        self.target[changed] = actions[sel]
        t = self.time[idx, None] + np.arange(1, cfg.physics_substeps + 1) * .001
        T = self.sp_T[idx, None]
        s = np.clip((t - self.sp_t0[idx, None]) / T, 0, 1)
        pos = (2 * s ** 3 - 3 * s * s + 1) * self.sp_p0[idx, None] + (s ** 3 - 2 * s * s + s) * T * self.sp_v0[idx, None] \
            + (-2 * s ** 3 + 3 * s * s) * self.sp_p1[idx, None]
        ctrl = inverse_kinematics(pos, np.zeros_like(pos))
        states, sensors = self.engine.rollout(self.model, self.data, self.state[idx], ctrl)
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(sensors)):
            raise ValueError('nonfinite physical trajectory')
        # rollout can return finite, repeated states after a MuJoCo warning.
        # A reset or frozen tail is a failed trajectory, never a task result.
        if not np.allclose(states[:, :, 0], t, rtol=0, atol=1e-10):
            raise ValueError('physical simulation reset or stopped advancing')
        puck_position = sensors[:, :, :2]
        if (np.any(np.linalg.norm(sensors[:, :, 3:5], axis=2) > 50)
                or np.any(np.abs(states[:, :, 8:]) > 100)
                or np.any(puck_position[:, :, 0] < -.1)
                or np.any(puck_position[:, :, 0] > cfg.table_length + .1)
                or np.any(np.abs(puck_position[:, :, 1]) > cfg.table_width / 2 + .1)):
            raise ValueError('physical trajectory exceeds safety bounds')
        self.state[idx] = states[:, -1]
        self.time[idx] += cfg.control_dt
        self.steps[idx] += 1
        # Sensor order: puck position/linear velocity, mallet position/velocity, touch.
        scored = np.any((sensors[:, :, 0] > cfg.goal_x) & (sensors[:, :, 3] > cfg.goal_vx), axis=1)
        contact = np.any(sensors[:, :, 12] > 0, axis=1)
        self.success[idx] |= scored
        self.hit[idx] |= contact
        sensor = sensors[:, -1]
        self.mx[idx], self.my[idx], self.mvx[idx] = sensor[:, 6], sensor[:, 7], sensor[:, 9]
        self.obs[idx] = sensor[:, [0, 1, 3, 4, 6, 7]]
        # Eq. 2 uses the instantaneous distance and velocity at the control tick.
        near = np.linalg.norm(sensor[:, :2] - sensor[:, 6:8], axis=1) <= .08
        reward = np.where(scored, cfg.terminal_reward,
                          np.where(near, cfg.contact_reward_scale * sensor[:, 3], cfg.step_penalty))
        self.done[idx] = self.success[idx] | (self.steps[idx] >= cfg.max_steps)
        return reward, self.done[idx].copy()
