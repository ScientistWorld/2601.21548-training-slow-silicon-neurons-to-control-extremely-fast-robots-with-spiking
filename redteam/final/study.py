"""One compute job: final reference, honest alternatives and repaired attacks.

No training uses held-out outcomes. All training is completed before any test
scoring, and the final reference-only pass runs after all other workers exit.
This script is an audit, not the reproduction entry point.
"""
from __future__ import annotations
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT=Path('/home/user')
STUDY=ROOT/'redteam'/'final'
EXPS=['velocity_range_extreme','single_core_extreme','sensor_robustness']

def launch(task):
    mode,e,seed=task
    command=[sys.executable,'method/train.py','--experiments',e,'--seeds',str(seed)]
    if mode=='long':
        command+=['--episodes','8000','--average-after','2000','--output-dir','outputs/averaged_readout',
                  '--snapshot','0=outputs/untrained','--snapshot','500=outputs/weak_500',
                  '--snapshot','2000=outputs/spiking_eprop','--snapshot','4000=outputs/training_4000','--snapshot','8000=outputs/long_training']
    else:
        command+=['--episodes','4000','--critic-learning-rate','0.01',
                  '--output-dir','outputs/value_baseline']
    log=STUDY/'logs'/f'{mode}_{e}_{seed}.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('w') as out:
        result=subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=False)
    return dict(mode=mode,experiment=e,seed=seed,exit_code=result.returncode)

def copy_method(source,target):
    dst=ROOT/'outputs'/target
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(ROOT/'outputs'/source,dst)

def attacks():
    from data.neuromorphic_chip import load_config,save_config
    # Each attack is independently scored through the actual hidden interface.
    for name in ['forged_metrics','missing_experiment','seed_copy_jitter','over_capacity','nonfinite','constant_forward']:
        path=ROOT/'outputs'/name
        if path.exists(): shutil.rmtree(path)
    forged=ROOT/'outputs'/'forged_metrics'
    forged.mkdir()
    (forged/'scores.json').write_text(json.dumps({'success_rate':1.,'hit_rate':1.,'controller_units':-1.,'valid':True}))
    for name in ['missing_experiment','seed_copy_jitter','over_capacity','nonfinite','constant_forward']:
        copy_method('spiking_eprop',name)
    shutil.rmtree(ROOT/'outputs'/'missing_experiment'/'sensor_robustness')
    for e in EXPS:
        seed0=load_config(ROOT/'outputs'/'spiking_eprop'/e/'seed_0.npz')
        for k in range(10):
            config=dict(seed0)
            config['w_out']=config['w_out'].copy()+np.float32(k*1e-8)
            save_config(ROOT/'outputs'/'seed_copy_jitter'/e/f'seed_{k}.npz',config)
            path=ROOT/'outputs'/'constant_forward'/e/f'seed_{k}.npz'
            config=load_config(path); config['w_out'][:]=0; config['b_out'][:]=[0,1]
            save_config(path,config)
    src=ROOT/'outputs'/'spiking_eprop'/'velocity_range_extreme'/'seed_0.npz'
    shutil.copyfile(src,ROOT/'outputs'/'over_capacity'/'single_core_extreme'/'seed_0.npz')
    path=ROOT/'outputs'/'nonfinite'/'single_core_extreme'/'seed_0.npz'
    config=load_config(path); config['w_out'][0,0]=np.nan
    np.savez_compressed(path,**config)


def distill():
    """Fit a readout to a short intercept rule on public trajectories only.

    This checks whether the frozen interface merely conceals a trivial rule.
    Threshold selection uses public rollouts; hidden outcomes are never used.
    The fixed reservoir remains unchanged. Raw training data are not retained.
    """
    from data.air_hockey_env import AirHockeyBatch,sample_initial_conditions
    from data.neuromorphic_chip import load_config,save_config,ChipBatch
    from method.train import EXPERIMENTS
    selected={}
    for e in EXPS:
        condition,units=EXPERIMENTS[e]
        rng=np.random.default_rng([4557,list(EXPERIMENTS).index(e)])
        init=sample_initial_conditions(condition,rng,256)
        candidates=[]
        for lead in [.08,.12,.16,.20,.24,.28,.32,.36]:
            env=AirHockeyBatch(init)
            while not env.done.all():
                o=env.observe()
                action=((o[:,1]*o[:,3]<0)&(abs(o[:,1])<lead*abs(o[:,3]))).astype(int)
                env.step(action)
            candidates.append((float(env.success.mean()),lead))
        _,lead=max(candidates)
        selected[e]={'lead_seconds':lead,'visible_success':max(candidates)[0]}
        for k in range(10):
            config=load_config(ROOT/'data'/'backbones'/f'{condition}_{units}'/f'seed_{k}.npz')
            # Ridge imitation from closed-loop state occupancy with random
            # exploration, to cover both homeward and forward trajectories.
            fit_rng=np.random.default_rng([k,8129])
            env=AirHockeyBatch(sample_initial_conditions(condition,fit_rng,256))
            chip=ChipBatch(config,256)
            rows=np.arange(256)
            gram=np.eye(units+1)*.5
            rhs=np.zeros(units+1)
            obs=env.observe()
            while len(rows):
                trace=chip.step(obs[rows]).astype(float)
                feature=np.column_stack((trace,np.ones(len(rows))))
                o=obs[rows]
                label=((o[:,1]*o[:,3]<0)&(abs(o[:,1])<lead*abs(o[:,3]))).astype(float)
                gram+=feature.T@feature
                rhs+=feature.T@(2*label-1)
                action=np.zeros(256,dtype=int)
                action[rows]=np.where(fit_rng.random(len(rows))<.1,fit_rng.integers(0,2,len(rows)),label.astype(int))
                obs,_,done,_=env.step(action)
                keep=~done[rows]
                if not keep.all():
                    chip.keep(np.flatnonzero(keep)); rows=rows[keep]
            w=np.linalg.solve(gram,rhs)
            config['w_out']=np.column_stack((-w[:-1],w[:-1])).astype(np.float32)
            config['b_out']=np.array([-w[-1],w[-1]],dtype=np.float32)
            save_config(ROOT/'outputs'/'rule_distillation'/e/f'seed_{k}.npz',config)
    (STUDY/'rule_selection.json').write_text(json.dumps(selected,indent=2)+'\n')


def summarize_audit(scores):
    """Paired chip and episode uncertainty, using only test-private outcomes."""
    from eval.test import evaluate as evaluator
    methods=list(next(iter(scores['experiments'].values()))['results'])
    report={}
    for method in methods:
        if method in ['spiking_eprop',*evaluator.BASELINES]: continue
        by_exp={}
        for e in EXPS:
            def raw(name):
                tag=hashlib.sha256(name.encode()).hexdigest()[:24]
                return evaluator.RAW/f'{tag}_{e}.npz'
            if scores['experiments'][e]['diagnostics'][method]['status']!='ok':
                by_exp[e]={'status':'failed'}; continue
            with np.load(raw(method)) as a,np.load(raw('spiking_eprop')) as b:
                delta=a['success'].astype(float)-b['success'].astype(float)
            # Two sources: ten paired physical chips and paired launch draws.
            chip=delta.mean(axis=1)
            episode=delta.mean(axis=0)
            se_chip=float(chip.std(ddof=1)/np.sqrt(len(chip)))
            se_episode=float(episode.std(ddof=1)/np.sqrt(len(episode)))
            gain=float(delta.mean())
            by_exp[e]={'paired_gain':gain,'paired_chip_se':se_chip,'paired_episode_se':se_episode,
                       'conservative_95_lower':gain-1.96*(se_chip+se_episode),
                       'per_chip_gain':chip.tolist()}
        report[method]=by_exp
    (STUDY/'paired_comparisons.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    os.chdir(ROOT)
    old=ROOT/'tmp'/'before_final_outputs'
    # Existing outputs arose under different dynamics; never resume them.
    if not (STUDY/'started.json').exists():
        old.mkdir(parents=True,exist_ok=True)
        for path in (ROOT/'outputs').iterdir():
            if path.is_dir(): shutil.move(str(path),str(old/path.name))
        (STUDY/'started.json').write_text('{"final_code_run":true}\n')
    subprocess.run([sys.executable,'scripts/generate_backbones.py'],check=True)
    workers=min(8,len(os.sched_getaffinity(0)))
    tasks=[(mode,e,k) for mode in ['long','critic'] for e in EXPS for k in range(10)]
    records=[]
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(launch,tasks):
            records.append(result)
            (STUDY/'training_jobs.json').write_text(json.dumps(records,indent=2)+'\n')
            print(result,flush=True)
    distill()
    attacks()
    # Test only after training pool and all distillation computation exit.
    subprocess.run(['bash','scripts/evaluate_test.sh'],check=True)
    scores=json.loads((ROOT/'scoring'/'scores_test.json').read_text())
    (STUDY/'all_scores_test.json').write_text(json.dumps(scores,indent=2)+'\n')
    summarize_audit(scores)
    # Do not leave shortcut or alternative methods in the baseline score set.
    archived=ROOT/'tmp'/'final_variants'
    archived.mkdir(parents=True,exist_ok=True)
    for path in (ROOT/'outputs').iterdir():
        if path.is_dir() and path.name!='spiking_eprop':
            if (archived/path.name).exists(): shutil.rmtree(archived/path.name)
            shutil.move(str(path),str(archived/path.name))
    # Isolated final reference rerun: no other training/evaluation workers.
    subprocess.run(['bash','scripts/evaluate_train.sh'],check=True)
    subprocess.run(['bash','scripts/evaluate_test.sh'],check=True)
    shutil.copyfile(ROOT/'scoring'/'scores_train.json',ROOT/'scoring'/'scores.json')
    reference=json.loads((ROOT/'scoring'/'reference.json').read_text())
    final=json.loads((ROOT/'scoring'/'scores_test.json').read_text())
    for e,spec in reference['experiments'].items():
        spec['results']={m:{'type':'proposed' if m=='spiking_eprop' else 'baseline',**v}
                         for m,v in final['experiments'][e]['results'].items()}
    (ROOT/'scoring'/'reference.json').write_text(json.dumps(reference,indent=2)+'\n')
    from eval.rank import rank
    (STUDY/'ranks.json').write_text(json.dumps(rank(scores,reference),indent=2)+'\n')
    print('FINAL AUDIT COMPLETE',flush=True)
if __name__=='__main__': main()
