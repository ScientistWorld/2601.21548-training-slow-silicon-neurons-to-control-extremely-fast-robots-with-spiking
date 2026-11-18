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
import tempfile
import numpy as np

ROOT=Path('/home/user')
STUDY=ROOT/'redteam'/'final'
EXPS=['velocity_range_extreme','single_core_extreme','sensor_robustness']

def launch(task):
    mode,e,seed=task
    command=[sys.executable,'method/train.py','--experiments',e,'--seeds',str(seed)]
    if mode!='long': raise ValueError(mode)
    command+=['--episodes','64000','--average-after','32000','--output-dir','outputs/averaged_readout',
              '--snapshot','0=outputs/untrained','--snapshot','2000=outputs/weak_2000',
              '--snapshot','32000=outputs/spiking_eprop','--snapshot','64000=outputs/long_training']
    log=STUDY/'logs_v2'/f'{mode}_{e}_{seed}.log'
    log.parent.mkdir(exist_ok=True)
    # Worker retries must preserve the last completed trial count and any
    # diagnostics from the previous process.
    with log.open('a') as out:
        out.write('\n--- training invocation (existing checkpoints retained) ---\n')
        out.flush()
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
            config=load_config(path); config['w_out'][:]=0; config['b_out'][:]=[0,100]
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
            for smoothing in [0.,.04,.08]:
                env=AirHockeyBatch(init)
                filtered=env.observe().copy()
                decay=np.exp(-.02/smoothing) if smoothing>0 else 0.
                while not env.done.all():
                    o=env.observe()
                    filtered=decay*filtered+(1-decay)*o
                    action=((filtered[:,1]*filtered[:,3]<0)&(abs(filtered[:,1])<lead*abs(filtered[:,3]))).astype(int)
                    env.step(action)
                candidates.append((float(env.success.mean()),lead,smoothing))
        _,lead,smoothing=max(candidates)
        selected[e]={'lead_seconds':lead,'smoothing_seconds':smoothing,'visible_success':max(candidates)[0],
                     'readout_logit_scale':32}
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
            filtered=obs.copy()
            decay=np.exp(-.02/smoothing) if smoothing>0 else 0.
            while len(rows):
                trace=chip.step(obs[rows]).astype(float)
                feature=np.column_stack((trace,np.ones(len(rows))))
                filtered[rows]=decay*filtered[rows]+(1-decay)*obs[rows]
                o=filtered[rows]
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
            config['w_out']=(32*np.column_stack((-w[:-1],w[:-1]))).astype(np.float32)
            config['b_out']=32*np.array([-w[-1],w[-1]],dtype=np.float32)
            save_config(ROOT/'outputs'/'rule_distillation'/e/f'seed_{k}.npz',config)
    (STUDY/'rule_selection.json').write_text(json.dumps(selected,indent=2)+'\n')


def summarize_audit(scores):
    """Paired chip and episode uncertainty, using only test-private outcomes."""
    from eval.test import evaluate as evaluator
    methods=list(next(iter(scores['experiments'].values()))['results'])
    report={}
    for method in methods:
        if method=='spiking_eprop': continue
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



def snapshot_histories():
    # These snapshots share the exact reference trajectory; retain the matching
    # online diagnostics instead of implying the averaged 64k log is the ref.
    for e in EXPS:
        for k in range(10):
            full=json.loads((ROOT/'outputs'/'averaged_readout'/e/f'training_seed_{k}.json').read_text())
            for name,episodes in [('spiking_eprop',32000),('weak_2000',2000),('long_training',64000)]:
                history=dict(full)
                for key in ['online_success','online_hit','episode_steps']:
                    history[key]=full[key][:episodes]
                history.pop('averaged_updates',None)
                history.pop('wall_seconds',None)
                history.update(method=name,episodes=episodes,
                    online_success_first200=float(np.mean([int(x) for x in history['online_success'][:200]])),
                    online_success_last200=float(np.mean([int(x) for x in history['online_success'][-200:]])))
                target=ROOT/'outputs'/name/e/f'training_seed_{k}.json'
                target.write_text(json.dumps(history)+'\n')


def scientific_checks(scores,reference):
    from eval.rank import rank
    ranks=rank(scores,reference)
    paired=json.loads((STUDY/'paired_comparisons.json').read_text())
    lower={}
    for method in ['long_training','averaged_readout']:
        bound=0.
        for e,spec in reference['experiments'].items():
            values=[r['success_rate'] for r in scores['experiments'][e]['results'].values()]
            scale=float(np.std(values))
            bound+=spec['weight']*paired[method][e]['conservative_95_lower']/scale if scale else 0.
        lower[method]=bound
    rules=['rule_distillation','intercept_rule']
    report={'ranks':ranks,'honest_paired_95_lower_rank_gain':lower,
            'honest_gain_exceeds_paired_noise':max(lower.values())>0.,
            'timing_rules_below_reference':all(ranks[m]<0 for m in rules),
            'weak_and_simple_below_reference':all(ranks[m]<0 for m in ['weak_2000','untrained','random_policy','always_forward']),
            'reference_at_bound':{e:spec['results']['spiking_eprop']['success_rate']==1.
                                  for e,spec in reference['experiments'].items()},
            'honest_variant_at_all_bounds':{m:all(scores['experiments'][e]['results'][m]['success_rate']==1.
                                                for e in EXPS) for m in lower}}
    report['passed']=(report['honest_gain_exceeds_paired_noise'] and report['timing_rules_below_reference']
                      and report['weak_and_simple_below_reference'] and not any(report['honest_variant_at_all_bounds'].values()))
    (STUDY/'scientific_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SCIENTIFIC CHECKS',report,flush=True)


def write_json(path,value):
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,
                prefix=f'.{path.name}.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(value,stream,indent=2)
            stream.write('\n')
        os.replace(temporary,path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def preflight():
    """Reuse completed checks only when their code and fixed assets match.

    This saves worker-retry time; it never caches training or test metrics.
    Raw outcome reuse has its separate exact-input validation.
    """
    def fingerprint():
        digest=hashlib.sha256(b'combined-audit-preflight-v1')
        paths=[]
        for folder in ['data','method','eval']:
            paths.extend((ROOT/folder).rglob('*.py'))
            paths.extend((ROOT/folder).rglob('*.xml'))
        for folder in ['data/backbones','eval/test/backbones']:
            for suffix in ['*.npz','*.json']:
                paths.extend((ROOT/folder).rglob(suffix))
        paths.extend(STUDY/name for name in ['check_resume.py','resume_scores.py',
                     'check_learning_resume.py','check_physics.py'])
        paths.append(ROOT/'scripts'/'generate_backbones.py')
        for path in sorted(set(paths)):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
        import mujoco
        digest.update((sys.version+np.__version__+mujoco.__version__).encode())
        return digest.hexdigest()
    marker=STUDY/'preflight_complete.json'
    if marker.exists() and json.loads(marker.read_text()).get('fingerprint')==fingerprint():
        print('preflight resume: source and fixed-asset fingerprints match',flush=True)
        return
    for script in ['redteam/final/check_resume.py',
                   'redteam/final/check_learning_resume.py',
                   'redteam/final/check_physics.py',
                   'scripts/generate_backbones.py']:
        subprocess.run([sys.executable,script],check=True)
    write_json(marker,{'fingerprint':fingerprint(),'checks_passed':True})


def main():
    os.chdir(ROOT)
    old=ROOT/'tmp'/'rejected_robot_setup_outputs'
    # Existing outputs arose under different dynamics; never resume them.
    if not (STUDY/'started.json').exists():
        old.mkdir(parents=True,exist_ok=True)
        for path in (ROOT/'outputs').iterdir():
            if path.is_dir(): shutil.move(str(path),str(old/path.name))
        (STUDY/'started.json').write_text('{"final_code_run":true}\n')
    preflight()
    if not (STUDY/'training_complete.json').exists():
        workers=min(8,len(os.sched_getaffinity(0)))
        tasks=[('long',e,k) for e in EXPS for k in range(10)]
        records=[]
        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            pending=[pool.submit(launch,task) for task in tasks]
            try:
                for completed in futures.as_completed(pending):
                    result=completed.result()
                    records.append(result)
                    write_json(STUDY/'training_jobs.json',records)
                    print(result,flush=True)
                    if result['exit_code']:
                        raise RuntimeError(f'training failed; inspect its log: {result}')
            except BaseException:
                for task in pending: task.cancel()
                raise
        snapshot_histories()
        distill()
        attacks()
        (STUDY/'training_complete.json').write_text('{"complete":true}\n')
    # Test only after training pool and all distillation computation exit.
    if not (STUDY/'audit_complete.json').exists():
        subprocess.run([sys.executable,'redteam/final/resume_scores.py','test'],check=True)
        scores=json.loads((ROOT/'scoring'/'scores_test.json').read_text())
        (STUDY/'all_scores_test.json').write_text(json.dumps(scores,indent=2)+'\n')
        summarize_audit(scores)
        (STUDY/'audit_complete.json').write_text('{"complete":true}\n')
    scores=json.loads((STUDY/'all_scores_test.json').read_text())
    # Do not leave shortcut or alternative methods in the baseline score set.
    archived=ROOT/'tmp'/'final_variants'
    archived.mkdir(parents=True,exist_ok=True)
    for path in (ROOT/'outputs').iterdir():
        if path.is_dir() and path.name!='spiking_eprop':
            if (archived/path.name).exists(): shutil.rmtree(archived/path.name)
            shutil.move(str(path),str(archived/path.name))
    # Isolated final reference rerun: no other training/evaluation workers.
    subprocess.run([sys.executable,'redteam/final/resume_scores.py','train'],check=True)
    subprocess.run([sys.executable,'redteam/final/resume_scores.py','test'],check=True)
    shutil.copyfile(ROOT/'scoring'/'scores_train.json',ROOT/'scoring'/'scores.json')
    reference=json.loads((ROOT/'scoring'/'reference.json').read_text())
    final=json.loads((ROOT/'scoring'/'scores_test.json').read_text())
    for e,spec in reference['experiments'].items():
        spec['results']={m:{'type':'proposed' if m=='spiking_eprop' else 'baseline',**v}
                         for m,v in final['experiments'][e]['results'].items()}
    reference['_setup_note']=reference['_setup_note'].replace('Local reference results are populated after the combined audit; zero placeholders are not measured results.','Local reference results below were recomputed from isolated rollouts after all training completed.')
    (ROOT/'scoring'/'reference.json').write_text(json.dumps(reference,indent=2)+'\n')
    scientific_checks(scores,reference)
    from eval.rank import rank
    (STUDY/'ranks.json').write_text(json.dumps(rank(scores,reference),indent=2)+'\n')
    subprocess.run([sys.executable,'validate.py'],check=True)
    print('FINAL AUDIT COMPLETE',flush=True)
if __name__=='__main__': main()
