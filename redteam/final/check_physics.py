"""Compute-node preflight for finite MuJoCo resets and contact stability.

Only public training draws are used. This is run once before chip calibration
and learning, in the same combined audit job.
"""
import json
from pathlib import Path
import numpy as np
from data.air_hockey_env import AirHockeyBatch, sample_initial_conditions


def main():
    report=[]
    for condition in ('speed_extreme','sensor_robustness'):
        for policy in ('random','alternating','forward','intercept'):
            rng=np.random.default_rng([44283,len(report)])
            env=AirHockeyBatch(sample_initial_conditions(condition,rng,512))
            peak_puck_speed=0.
            for step in range(env.cfg.max_steps):
                if env.done.all(): break
                obs=env.observe()
                if policy=='random': action=rng.integers(0,2,env.batch)
                elif policy=='alternating': action=(np.arange(env.batch)+step)%2
                elif policy=='forward': action=np.ones(env.batch,dtype=int)
                else:
                    lead=np.linspace(.08,.40,env.batch)
                    action=((obs[:,1]*obs[:,3]<0)&(abs(obs[:,1])<lead*abs(obs[:,3]))).astype(int)
                env.step(action)
                peak_puck_speed=max(peak_puck_speed,float(np.linalg.norm(env.state[:,6:8],axis=1).max()))
            if peak_puck_speed>50:
                raise ValueError(f'unphysical puck speed: {peak_puck_speed}')
            report.append(dict(condition=condition,policy=policy,episodes=env.batch,
                               success=float(env.success.mean()),peak_puck_speed=peak_puck_speed))
            print('physics preflight',report[-1],flush=True)

    # Finite arrays cannot hide a frozen rollout. Check the failure path too.
    class FrozenEngine:
        def rollout(self,model,data,initial,control):
            return np.repeat(initial[:,None,:],20,axis=1),np.zeros((len(initial),20,13))
    env=AirHockeyBatch(sample_initial_conditions('speed_extreme',np.random.default_rng(72),1))
    env.engine=FrozenEngine()
    try:
        env.step([0])
    except ValueError as exc:
        if 'stopped advancing' not in str(exc): raise
    else:
        raise AssertionError('finite stalled rollout was accepted')
    Path('redteam/final/physics_preflight.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
