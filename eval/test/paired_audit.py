"""Paired audit uncertainty from raw held-out outcomes, without rerunning models.

Resample chip identities and whole episode columns, preserving comparisons
between methods and all within-episode dependence. Bounded, skewed chip gains
need not have a Gaussian sampling distribution. Rank uses the measured cohort
scales, kept fixed during this uncertainty calculation.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
from eval.test import evaluate

def bootstrap_deltas(deltas,weights,replicates=4096,seed=510_923):
    methods,chips,episodes=deltas[0].shape
    if any(d.shape!=(methods,chips,episodes) for d in deltas):
        raise ValueError('inconsistent paired dimensions')
    rng=np.random.default_rng(seed)
    chip_counts=rng.multinomial(chips,np.full(chips,1/chips),size=replicates)/chips
    samples=np.zeros((methods,replicates))
    per_chip=np.zeros((methods,chips))
    for delta,weight in zip(deltas,weights):
        per_chip+=weight*delta.mean(axis=2)
        for start in range(0,replicates,64):
            stop=min(start+64,replicates)
            counts=rng.multinomial(episodes,np.full(episodes,1/episodes),size=stop-start)/episodes
            # Each method sees the same resampled chips and episode draws.
            means=(delta.reshape(methods*chips,episodes)@counts.T).reshape(methods,chips,stop-start)
            samples[:,start:stop]+=weight*np.einsum('mcb,bc->mb',means,chip_counts[start:stop])
    intervals=np.quantile(samples,[.025,.975],axis=1).T
    report=[]
    for index in range(methods):
        values=per_chip[index]
        nonzero=values[values!=0]
        positive=int(np.sum(nonzero>0))
        p=sum(math.comb(len(nonzero),k) for k in range(positive,len(nonzero)+1))/(2**len(nonzero))
        report.append({'paired_rank_gain':float(values.mean()),
                       'paired_bootstrap_95_lower':float(intervals[index,0]),
                       'paired_bootstrap_95_upper':float(intervals[index,1]),
                       'positive_chips':positive,'nonzero_chips':len(nonzero),
                       'one_sided_sign_p':float(p),'per_chip_rank_gain':values.tolist(),
                       'bootstrap_replicates':replicates})
    return report

def compare(scores,reference,methods):
    def success(name,experiment):
        tag=hashlib.sha256(name.encode()).hexdigest()[:24]
        with np.load(evaluate.RAW/f'{tag}_{experiment}.npz',allow_pickle=False) as raw:
            result=raw['success'].astype(float)
        if result.shape!=(evaluate.N_CONTROLLERS,evaluate.N_EPISODES) or not np.all(np.isin(result,[0,1])):
            raise ValueError('invalid paired outcomes')
        return result
    deltas,weights=[],[]
    for experiment,spec in reference['experiments'].items():
        scale=float(np.std([row['success_rate'] for row in scores['experiments'][experiment]['results'].values()]))
        baseline=success('spiking_eprop',experiment)
        deltas.append(np.stack([success(method,experiment)-baseline for method in methods]))
        definition=spec['metrics']['success_rate']
        direction=1 if definition['higher_is_better'] else -1
        weights.append(spec['weight']*abs(definition['coefficient'])*direction/scale if scale else 0.)
    return dict(zip(methods,bootstrap_deltas(deltas,weights)))
