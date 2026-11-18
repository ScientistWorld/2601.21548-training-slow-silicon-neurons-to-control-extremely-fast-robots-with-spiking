"""Small compute-job check of paired bootstrap direction and cancellation."""
import numpy as np
from eval.test.paired_audit import bootstrap_deltas

def main():
    rng=np.random.default_rng(44)
    improvement=rng.uniform(.01,.2,(10,128))
    deltas=np.stack([improvement,np.zeros_like(improvement),-improvement])
    report=bootstrap_deltas([deltas,deltas],[.3,.7],replicates=256)
    assert report[0]['paired_bootstrap_95_lower']>0
    assert report[1]['paired_bootstrap_95_lower']==report[1]['paired_bootstrap_95_upper']==0
    assert report[2]['paired_bootstrap_95_upper']<0
    assert report[0]['positive_chips']==10 and report[0]['one_sided_sign_p']==1/1024
    assert np.isclose(report[0]['paired_bootstrap_95_lower'],-report[2]['paired_bootstrap_95_upper'])
    print('paired resampling direction, shared draws and zero-gain checks passed',flush=True)

if __name__=='__main__': main()
