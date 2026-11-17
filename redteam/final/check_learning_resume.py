"""Compute-node check: interruption preserves Eq. 1 updates exactly."""
from pathlib import Path
import tempfile
import numpy as np
from method import spiking_eprop_rl as learning


def main():
    cfg=learning.EpropConfig(hidden_neurons=256,extra={'average_after':8})
    expected,history=learning.train_controller('speed_extreme',0,16,cfg)
    Path('tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir='tmp') as directory:
        checkpoint=Path(directory)/'training.npz'
        original=learning.EpropReadoutLearner.update
        calls=0
        class WorkerDrain(Exception): pass
        def interrupt(self,trajectories):
            nonlocal calls
            calls+=1
            if calls==6: raise WorkerDrain()
            return original(self,trajectories)
        learning.EpropReadoutLearner.update=interrupt
        try:
            learning.train_controller('speed_extreme',0,16,cfg,
                resume_path=checkpoint,checkpoint_every_trials=2)
        except WorkerDrain:
            pass
        else:
            raise AssertionError('interruption was not triggered')
        finally:
            learning.EpropReadoutLearner.update=original
        assert checkpoint.exists()
        resumed,resumed_history=learning.train_controller('speed_extreme',0,16,cfg,
            resume_path=checkpoint,checkpoint_every_trials=2)
        assert history==resumed_history
        for key in expected:
            assert np.array_equal(expected[key],resumed[key]),key
    print('interrupted/uninterrupted training is bit-identical',flush=True)

if __name__=='__main__': main()
