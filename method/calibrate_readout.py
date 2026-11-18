"""Public-validation calibration of learned readouts; an honest audit variant.

The reference is unchanged Eq. 1 training. This alternative chooses one global
readout blend, inverse temperature and log-odds offset per experiment, using
only fresh public simulator launches. No hidden evaluator is imported.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from data.air_hockey_env import AirHockeyBatch, sample_initial_conditions
from data.neuromorphic_chip import ChipBatch, load_config, save_config

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS={'velocity_range_extreme':'speed_extreme',
             'single_core_extreme':'speed_extreme',
             'sensor_robustness':'sensor_robustness'}
PUBLIC_SEED=739_204_117
CANDIDATES=[(blend,temperature,offset) for blend in (0.,.5,1.)
            for temperature in (1.,2.,4.,8.) for offset in (0.,-2.,2.)]

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,
                prefix=f'.{path.name}.',delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(value,stream)
            stream.write('\n')
        os.replace(temporary,path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)

def transformed(reference,average,parameters):
    blend,temperature,offset=parameters
    result=dict(reference)
    for key in ('w_out','b_out'):
        result[key]=(temperature*((1-blend)*reference[key]+blend*average[key])).astype(np.float32)
    result['b_out']=result['b_out']+np.array([-offset/2,offset/2],dtype=np.float32)
    return result

def public_success(config,initial,experiment,seed):
    successes=[]
    for start in range(0,len(initial),128):
        part=initial[start:start+128]
        env=AirHockeyBatch(part)
        chip=ChipBatch(config,len(part))
        rows=np.arange(len(part))
        observation=env.observe()
        rng=np.random.default_rng([PUBLIC_SEED,experiment,seed,start,51])
        draws=rng.random((len(part),100))
        tick=0
        while rows.size:
            logits=chip.scores(chip.step(observation[rows])).astype(np.float64)
            logits-=logits.max(axis=1,keepdims=True)
            probability=np.exp(logits)
            probability/=probability.sum(axis=1,keepdims=True)
            actions=np.zeros(len(part),dtype=int)
            actions[rows]=(draws[rows,tick]<probability[:,1]).astype(int)
            observation,_,done,info=env.step(actions)
            tick+=1
            keep=~done[rows]
            if not keep.all():
                chip.keep(np.flatnonzero(keep))
                rows=rows[keep]
        successes.extend(info['success'].tolist())
    return float(np.mean(successes))

def validate_chip(task):
    reference_root,average_root,output_root,experiment,seed,count=task
    reference_path=reference_root/experiment/f'seed_{seed}.npz'
    average_path=average_root/experiment/f'seed_{seed}.npz'
    import mujoco
    digest=hashlib.sha256((np.__version__+mujoco.__version__+str(count)).encode())
    for path in [Path(__file__),ROOT/'data/air_hockey_env.py',
                 ROOT/'data/neuromorphic_chip.py',ROOT/'data/robot/air_hockey.xml',
                 reference_path,average_path]:
        digest.update(path.read_bytes())
    fingerprint=digest.hexdigest()
    checkpoint=output_root/'public_validation'/experiment/f'seed_{seed}.json'
    rates=[]
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if saved.get('fingerprint')==fingerprint:
            rates=saved['rates']
    index=list(EXPERIMENTS).index(experiment)
    initial=sample_initial_conditions(EXPERIMENTS[experiment],
                np.random.default_rng([PUBLIC_SEED,index]),count)
    reference=load_config(reference_path)
    average=load_config(average_path)
    for candidate in CANDIDATES[len(rates):]:
        rates.append(public_success(transformed(reference,average,candidate),initial,index,seed))
        write_json(checkpoint,{'fingerprint':fingerprint,'rates':rates})
        print('public calibration',experiment,seed,len(rates),len(CANDIDATES),rates[-1],flush=True)
    return experiment,seed,rates

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-root',type=Path,default=ROOT/'outputs/spiking_eprop')
    parser.add_argument('--average-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/public_calibrated')
    parser.add_argument('--sharpened-root',type=Path,default=ROOT/'outputs/sharpened_readout')
    parser.add_argument('--validation-episodes',type=int,default=256)
    args=parser.parse_args()
    tasks=[(args.reference_root,args.average_root,args.output_root,e,k,args.validation_episodes)
           for e in EXPERIMENTS for k in range(10)]
    rates={e:[] for e in EXPERIMENTS}
    with ProcessPoolExecutor(max_workers=min(8,len(os.sched_getaffinity(0)))) as pool:
        for experiment,seed,values in pool.map(validate_chip,tasks):
            rates[experiment].append((seed,values))
    selected={}
    for experiment in EXPERIMENTS:
        pooled=np.mean([values for _,values in sorted(rates[experiment])],axis=0)
        best=int(np.argmax(pooled))
        selected[experiment]={'parameters':CANDIDATES[best],
             'public_success':float(pooled[best]),'public_reference_success':float(pooled[0]),
             'public_episodes_per_chip':args.validation_episodes,'candidate_count':len(CANDIDATES)}
        for seed in range(10):
            reference=load_config(args.reference_root/experiment/f'seed_{seed}.npz')
            average=load_config(args.average_root/experiment/f'seed_{seed}.npz')
            save_config(args.output_root/experiment/f'seed_{seed}.npz',
                        transformed(reference,average,CANDIDATES[best]))
            save_config(args.sharpened_root/experiment/f'seed_{seed}.npz',
                        transformed(reference,average,(0.,8.,0.)))
    write_json(args.output_root/'selection.json',selected)
    print('PUBLIC CALIBRATION COMPLETE',selected,flush=True)

if __name__=='__main__': main()
