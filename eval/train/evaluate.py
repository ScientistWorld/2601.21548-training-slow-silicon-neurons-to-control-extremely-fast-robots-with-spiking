"""Visible validation scoring. Independent of the hidden evaluator.

Only raw bounded readout arrays cross the submission interface. No method
code runs, so the hidden simulator and future noisy observations cannot be
queried by private search. All methods see the same episodes and fixed chips.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from data.neuromorphic_chip import ChipBatch, ConfigError, controller_units, load_config
from data.air_hockey_env import AirHockeyBatch, sample_initial_conditions
ROOT=Path(__file__).resolve().parents[2]
OUTPUTS=ROOT/'outputs'
SCORES=ROOT/'scoring'/'scores_train.json'
RAW=Path(__file__).parent/'raw'
BACKBONES=ROOT/'data'/'backbones'
EPISODE_SEED=202_606_01
EXPERIMENTS={
    'velocity_range_extreme':('speed_extreme',1020),
    'single_core_extreme':('speed_extreme',256),
    'sensor_robustness':('sensor_robustness',1020),
}
N_CONTROLLERS=10
N_EPISODES=4096
BATCH=128
BASELINES=('random_policy','always_forward','intercept_rule')
METRICS=('success_rate','seed_worst_success','hit_rate','controller_units')

def worst(experiment):
    return dict(success_rate=0.,seed_worst_success=0.,hit_rate=0.,
                controller_units=float(EXPERIMENTS[experiment][1]))

def held_out_episodes(experiment):
    rng=np.random.default_rng([EPISODE_SEED,list(EXPERIMENTS).index(experiment)])
    return sample_initial_conditions(EXPERIMENTS[experiment][0],rng,N_EPISODES)

def _run(init,choose):
    env=AirHockeyBatch(init)
    rows=np.arange(env.batch)
    obs=env.observe()
    state=choose(None,rows,None)
    while rows.size:
        actions=np.zeros(env.batch,dtype=int)
        selected=np.asarray(choose(obs[rows],rows,state))
        if selected.shape!=(len(rows),) or not np.all(np.isin(selected,[0,1])):
            raise ConfigError('invalid actions')
        actions[rows]=selected
        obs,_,done,info=env.step(actions)
        keep=~done[rows]
        if not np.all(keep):
            rows=rows[keep]
            if state is not None and hasattr(state,'keep'):
                state.keep(np.flatnonzero(keep))
    return info['success'].copy(),info['hit'].copy()

def rollout_controller(config,init):
    successes,hits=[],[]
    for start in range(0,len(init),BATCH):
        part=init[start:start+BATCH]
        # Common random numbers: every controller sees the same action draws
        # at a given episode/tick, as well as the same physical/noise inputs.
        random=np.random.default_rng([EPISODE_SEED,int(part[0,3]),674383])
        draws=random.random((len(part),100))
        tick=0
        def choose(obs,rows,chip):
            nonlocal tick
            if obs is None: return ChipBatch(config,len(part))
            logits=chip.scores(chip.step(obs)).astype(np.float64)
            logits-=logits.max(axis=1,keepdims=True)
            probabilities=np.exp(logits)
            probabilities/=probabilities.sum(axis=1,keepdims=True)
            selected=(draws[rows,tick]<probabilities[:,1]).astype(int)
            tick+=1
            return selected
        s,h=_run(part,choose)
        successes.append(s); hits.append(h)
    return np.concatenate(successes),np.concatenate(hits)

def rollout_baseline(name,init,stream,experiment,lead=.15,smoothing=0.):
    rng=np.random.default_rng([EPISODE_SEED,list(EXPERIMENTS).index(experiment),stream,99])
    filtered=np.zeros((len(init),6))
    initialized=False
    decay=np.exp(-.02/smoothing) if smoothing>0 else 0.
    def choose(obs,rows,state):
        nonlocal initialized
        if obs is None: return None
        if not initialized:
            filtered[rows]=obs
            initialized=True
        else:
            filtered[rows]=decay*filtered[rows]+(1-decay)*obs
        obs=filtered[rows]
        if name=='random_policy': return rng.integers(0,2,len(rows))
        if name=='always_forward': return np.ones(len(rows),dtype=int)
        # Fixed analytic intercept-time rule, no fitting on held-out outcomes.
        # A fixed bank below covers the original hand rule and wider leads.
        travel=np.clip((obs[:,0]-obs[:,4])/2.,0.,.4)
        timing=lead if lead>=0 else travel+.06
        return ((obs[:,1]*obs[:,3]<0)&(abs(obs[:,1])<timing*abs(obs[:,3]))).astype(int)
    return _run(init,choose)

def load_controllers(method_dir,experiment):
    condition,budget=EXPERIMENTS[experiment]
    configs=[]
    for k in range(N_CONTROLLERS):
        config=load_config(method_dir/experiment/f'seed_{k}.npz')
        fixed=load_config(BACKBONES/f'{condition}_{budget}'/f'seed_{k}.npz')
        if set(config)!=set(fixed): raise ConfigError('configuration fields differ from fixed instance')
        for key in fixed:
            if key not in ('w_out','b_out') and not np.array_equal(config[key],fixed[key]):
                raise ConfigError(f'seed_{k}: {key} differs from assigned chip')
        if controller_units(config)!=budget: raise ConfigError('wrong unit count')
        configs.append(config)
    return configs

def evaluate_method_experiment(method_dir,experiment):
    configs=load_controllers(method_dir,experiment)
    init=held_out_episodes(experiment)
    s,h=zip(*(rollout_controller(c,init) for c in configs))
    return dict(success=np.stack(s),hit=np.stack(h),units=np.array([controller_units(c) for c in configs]))

def evaluate_baseline(name,experiment):
    init=held_out_episodes(experiment)
    # Deterministic rules need only one rollout; random has ten fixed streams.
    if name=='intercept_rule':
        # Conservative comparator: best global member of a predeclared rule
        # bank. No per-episode oracle, no method-selected threshold.
        candidates=[rollout_baseline(name,init,0,experiment,lead,smoothing)
                    for lead in [.08,.12,.16,.20,.24,.28,.32,.36,-1.]
                    for smoothing in [0.,.04,.08]]
        results=[max(candidates,key=lambda result:float(result[0].mean()))]
    else:
        results=[rollout_baseline(name,init,k,experiment) for k in range(10 if name=='random_policy' else 1)]
    if len(results)==1: results=results*10
    s,h=zip(*results)
    return dict(success=np.stack(s),hit=np.stack(h),units=np.zeros(10))

def summarize(outcome):
    for key in ('success','hit'):
        a=np.asarray(outcome[key])
        if a.shape!=(N_CONTROLLERS,N_EPISODES) or not np.all(np.isin(a,[0,1])):
            raise ConfigError('invalid outcome shape/value')
    s=np.asarray(outcome['success'],dtype=float)
    return dict(success_rate=float(s.mean()),seed_worst_success=float(s.mean(axis=1).min()),
                hit_rate=float(np.asarray(outcome['hit']).mean()),controller_units=float(max(outcome['units'])))

def discover_methods():
    # Include dot/underscore directories too: missing or malformed is failure.
    names={p.name for p in OUTPUTS.iterdir() if p.is_dir()} if OUTPUTS.exists() else set()
    reference=json.loads((ROOT/'scoring'/'reference.json').read_text())
    for exp in reference['experiments'].values():
        names.update(m for m,r in exp['results'].items() if r.get('type')=='proposed')
    return sorted(names-set(BASELINES))

def score_task(task):
    method,experiment,is_baseline=task
    try:
        outcome=evaluate_baseline(method,experiment) if is_baseline else evaluate_method_experiment(OUTPUTS/method,experiment)
        metrics=summarize(outcome)
        RAW.mkdir(parents=True,exist_ok=True)
        # Answers and paired outcomes never leave this split's private area.
        import hashlib
        tag=hashlib.sha256(method.encode()).hexdigest()[:24]
        np.savez_compressed(RAW/f'{tag}_{experiment}.npz',**outcome)
        diag={'status':'ok','per_controller_success':outcome['success'].mean(axis=1).tolist()}
    except Exception as exc:
        metrics=worst(experiment)
        diag={'status':'failed','reason':f'{type(exc).__name__}: {exc}'}
    return method,experiment,metrics,diag

def main():
    SCORES.parent.mkdir(parents=True,exist_ok=True)
    methods=discover_methods()
    all_results={e:{'results':{m:worst(e) for m in methods+list(BASELINES)},'diagnostics':{}} for e in EXPERIMENTS}
    for method in methods+list(BASELINES):
        rows=[score_task((method,e,method in BASELINES)) for e in EXPERIMENTS]
        # Any malformed experiment fails the entire submission. It cannot use
        # selective missing outputs to avoid a difficult scored experiment.
        failed=any(d['status']!='ok' for _,_,_,d in rows)
        for _,e,m,d in rows:
            if failed:
                m=worst(e)
                d={**d,'status':'failed','reason':d.get('reason','another experiment failed')}
            all_results[e]['results'][method]=m
            all_results[e]['diagnostics'][method]=d
            print(method,e,m,d.get('reason',''),flush=True)
        target=SCORES.with_suffix('.tmp')
        target.write_text(json.dumps({'experiments':all_results},indent=2)+'\n')
        target.replace(SCORES)
    return 0
if __name__=='__main__': raise SystemExit(main())
