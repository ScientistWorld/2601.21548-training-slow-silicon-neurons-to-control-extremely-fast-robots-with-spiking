"""Diagnostic cohort rank implementing the gym's metric normalization.

Only per-experiment results in scores_{train,test}.json are inputs. A failed
method has 0 in every scored metric of every experiment. Each metric's
improvement over the reference is taken in its better direction, multiplied by
the absolute coefficient and the experiment weight, and divided by the metric's
population standard deviation across the ranked cohort; identical columns
contribute zero. This helper does not write scored numbers.
"""
import math
import numpy as np

def rank(scores,reference):
    names=sorted({m for e in scores['experiments'].values() for m in e['results']})
    totals={m:0. for m in names}
    for e,spec in reference['experiments'].items():
        results=scores['experiments'][e]['results']
        for metric,definition in spec['metrics'].items():
            coefficient=abs(definition['coefficient'])
            if not coefficient: continue
            values=np.array([results[m][metric] for m in names],dtype=float)
            if not np.all(np.isfinite(values)): raise ValueError('nonfinite ranked value')
            scale=float(values.std())
            if scale==0: continue
            ref=reference['experiments'][e]['results']['spiking_eprop'][metric]
            direction=1 if definition['higher_is_better'] else -1
            for m,value in zip(names,values):
                totals[m]+=spec['weight']*coefficient*direction*(value-ref)/scale
    return dict(sorted(totals.items(),key=lambda kv:-kv[1]))
