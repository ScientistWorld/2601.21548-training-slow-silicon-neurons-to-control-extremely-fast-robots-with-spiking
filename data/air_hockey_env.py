"""Public air-hockey environment: torque-limited planar robot in MuJoCo.

The six observations, two task-space cubic primitives, 50 Hz decisions,
1 kHz physics, reward (Eq. 2), and success event follow the paper. The robot
geometry is an explicitly documented software substitute; see robot/README.md.
No labels, test instances, fitted policies, or test seeds are stored here.
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

CONDITIONS = {
    "stationary_1m": {"puck_x": (1.,1.), "speed": (0.,0.)},
    "constant_speed": {"puck_x": (1.,1.), "speed": (1.2,1.2)},
    "variable_speed": {"puck_x": (1.,1.), "speed": (1.,1.5)},
    "random_position_speed": {"puck_x": (.95,1.05), "speed": (1.,1.5)},
    "speed_narrow": {"puck_x": (1.,1.), "speed": (.7,.9)},
    "speed_medium": {"puck_x": (1.,1.), "speed": (.7,1.2)},
    "speed_wide": {"puck_x": (1.,1.), "speed": (.7,1.5)},
    "speed_extreme": {"puck_x": (1.,1.), "speed": (.5,2.)},
    # Robustness extension motivated by the paper's future-work discussion:
    # richer launches, noisy state estimation, and two control ticks latency.
    "sensor_robustness": {"puck_x": (.80,1.35), "speed": (.5,2.),
                          "angle": .30, "noise": .035, "latency": 2},
}

def sample_initial_conditions(condition, rng, n):
    spec = CONDITIONS[condition]
    return np.column_stack((rng.uniform(*spec["puck_x"],n), rng.uniform(*spec["speed"],n),
        rng.uniform(-spec.get("angle",0), 0,n),
        rng.integers(0,2**32-1,n), np.full(n,spec.get("noise",0)),
        np.full(n,spec.get("latency",0))))

def inverse_kinematics(x, y):
    wx = np.asarray(x) - .4
    y = np.asarray(y)
    c = np.clip((wx*wx+y*y-.75**2-.55**2)/(2*.75*.55), -.99999,.99999)
    q2 = np.arccos(c)
    q1 = np.arctan2(y,wx) - np.arctan2(.55*np.sin(q2), .75+.55*c)
    return np.clip(np.stack((q1,q2,-q1-q2),axis=-1),-3.05,3.05)

@lru_cache(maxsize=1)
def _physics():
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / 'robot' / 'air_hockey.xml'))
    return model, mujoco.MjData(model), rollout.Rollout(nthread=0)

class AirHockeyBatch:
    def __init__(self, init, config=None):
        self.cfg = config or AirHockeyConfig()
        init = np.asarray(init, dtype=np.float64)
        if init.ndim != 2 or init.shape[1] not in (2,6):
            raise ValueError('initial conditions must have 2 or 6 columns')
        if init.shape[1] == 2:
            init = np.column_stack((init,np.zeros((len(init),4))))
        self.batch = len(init)
        self.model, self.data, self.engine = _physics()
        self.state = np.zeros((self.batch,1+self.model.nq+self.model.nv))
        # FULLPHYSICS: time, qpos(puck x,y,robot angles), qvel.
        self.state[:,1] = init[:,0]
        self.state[:,2] = np.where(init[:,1]==0,0,self.cfg.launch_y)
        self.state[:,3:6] = inverse_kinematics(np.full(self.batch,self.cfg.home_x),np.zeros(self.batch))
        self.state[:,6] = init[:,1]*np.sin(init[:,2])
        self.state[:,7] = -init[:,1]*np.cos(init[:,2])
        self.time = np.zeros(self.batch)
        self.steps = np.zeros(self.batch,dtype=int)
        self.done = np.zeros(self.batch,dtype=bool)
        self.success = self.done.copy()
        self.hit = self.done.copy()
        self.target = np.zeros(self.batch,dtype=int)
        self.sp_t0 = np.zeros(self.batch)
        self.sp_p0 = np.full(self.batch,self.cfg.home_x)
        self.sp_v0 = np.zeros(self.batch)
        self.sp_p1 = self.sp_p0.copy()
        self.sp_T = np.ones(self.batch)
        self.mx = self.sp_p0.copy()
        self.my = np.zeros(self.batch)
        self.mvx = np.zeros(self.batch)
        self._true_obs = np.column_stack((self.state[:,1:3],self.state[:,6:8],self.mx,self.my))
        self._latency = init[:,5].astype(int)
        if np.any(self._latency < 0) or np.any(self._latency > 10):
            raise ValueError("observation latency must be 0..10 control ticks")
        self._history = np.repeat(self._true_obs[:,None,:],int(self._latency.max())+1,axis=1)
        self._noise = np.zeros((self.batch,self.cfg.max_steps+1,6))
        for i in range(self.batch):
            rng = np.random.default_rng(int(init[i,3]))
            scale = np.array([1,1,10,10,1,1])*init[i,4]
            self._noise[i] = rng.normal(size=(self.cfg.max_steps+1,6))*scale

    def observe(self):
        obs = self._history[np.arange(self.batch),self._history.shape[1]-1-self._latency]
        return obs + self._noise[np.arange(self.batch),self.steps]

    def step(self, actions):
        cfg = self.cfg
        actions = np.asarray(actions,dtype=int).reshape(-1)
        if len(actions)!=self.batch or np.any((actions<0)|(actions>1)):
            raise ValueError('one binary action per episode is required')
        idx = np.flatnonzero(~self.done)
        reward = np.zeros(self.batch)
        if not len(idx):
            return self.observe(),reward,self.done.copy(),self._info()
        changed = idx[actions[idx]!=self.target[idx]]
        self.sp_p0[changed] = self.mx[changed]
        self.sp_v0[changed] = self.mvx[changed]
        self.sp_p1[changed] = np.where(actions[changed],cfg.forward_x,cfg.home_x)
        self.sp_T[changed] = np.clip(abs(self.sp_p1[changed]-self.mx[changed])/cfg.mallet_mean_speed,
                                   cfg.min_primitive_s,cfg.max_primitive_s)
        self.sp_t0[changed] = self.time[changed]
        self.target[changed] = actions[changed]
        t = self.time[idx,None]+np.arange(1,cfg.physics_substeps+1)*.001
        T = self.sp_T[idx,None]
        s = np.clip((t-self.sp_t0[idx,None])/T,0,1)
        pos = (2*s**3-3*s*s+1)*self.sp_p0[idx,None] + (s**3-2*s*s+s)*T*self.sp_v0[idx,None] \
              +(-2*s**3+3*s*s)*self.sp_p1[idx,None]
        ctrl = inverse_kinematics(pos,np.zeros_like(pos))
        states,sensors = self.engine.rollout(self.model,self.data,self.state[idx],ctrl)
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(sensors)):
            raise ValueError('nonfinite physical trajectory')
        # rollout can return finite, repeated states after a MuJoCo warning.
        # A reset or frozen tail is a failed trajectory, never a task result.
        if not np.allclose(states[:,:,0],t,rtol=0,atol=1e-10):
            raise ValueError('physical simulation reset or stopped advancing')
        # Loose safety bounds catch numerical explosions before a spurious
        # goal can terminate an episode and conceal the next solver warning.
        puck_position=sensors[:,:,:2]
        if (np.any(np.linalg.norm(sensors[:,:,3:5],axis=2)>50)
                or np.any(np.abs(states[:,:,8:])>100)
                or np.any(puck_position[:,:,0]<-.1)
                or np.any(puck_position[:,:,0]>cfg.table_length+.1)
                or np.any(np.abs(puck_position[:,:,1])>cfg.table_width/2+.1)):
            raise ValueError('physical trajectory exceeds safety bounds')
        self.state[idx] = states[:,-1]
        self.time[idx] += cfg.control_dt
        self.steps[idx] += 1
        # Sensor order: puck position/linear velocity, mallet position/velocity,touch.
        scored = np.any((sensors[:,:,0]>cfg.goal_x)&(sensors[:,:,3]>cfg.goal_vx),axis=1)
        contact = np.any(sensors[:,:,12]>0,axis=1)
        self.success[idx] |= scored
        self.hit[idx] |= contact
        sensor = sensors[:,-1]
        self.mx[idx],self.my[idx],self.mvx[idx] = sensor[:,6],sensor[:,7],sensor[:,9]
        self._true_obs[idx] = sensor[:,[0,1,3,4,6,7]]
        self._history[idx,:-1] = self._history[idx,1:]
        self._history[idx,-1] = self._true_obs[idx]
        # Eq. 2 uses the instantaneous distance and velocity at the control tick.
        near = np.linalg.norm(sensor[:,:2]-sensor[:,6:8],axis=1) <= .08
        reward[idx] = np.where(scored,cfg.terminal_reward,
                             np.where(near,cfg.contact_reward_scale*sensor[:,3],cfg.step_penalty))
        self.done[idx] = self.success[idx] | (self.steps[idx]>=cfg.max_steps)
        return self.observe(),reward,self.done.copy(),self._info()

    def _info(self):
        return dict(success=self.success.copy(),hit=self.hit.copy(),steps=self.steps.copy())

class AirHockeyInterceptionEnv:
    def __init__(self,condition='random_position_speed',seed=0,config=None):
        self.condition,self.config = condition,config
        self.rng = np.random.default_rng(seed)
    def reset(self):
        self._batch = AirHockeyBatch(sample_initial_conditions(self.condition,self.rng,1),self.config)
        return self._batch.observe()[0]
    def step(self,action):
        obs,r,d,info = self._batch.step([action])
        return obs[0],float(r[0]),bool(d[0]),{k:v[0].item() for k,v in info.items()}
