"""Build public calibrated chip assets, once, on a compute node.

Random behaviour and chip calibration use no task answers or held-out seeds.
Generation calls the reproduction's initializer, not its training procedure.
The resulting small assets are tracked; test receives immutable copies.
"""
import json
import shutil
from pathlib import Path
import numpy as np
from method.spiking_eprop_rl import EpropConfig, build_reservoir
from data.neuromorphic_chip import load_config, save_config

ROOT=Path(__file__).resolve().parents[1]
PLAN=[('speed_extreme',1020),('speed_extreme',256),('sensor_robustness',1020)]
def main():
    for condition,units in PLAN:
        for seed in range(10):
            target=ROOT/'data'/'backbones'/f'{condition}_{units}'/f'seed_{seed}.npz'
            if not target.exists():
                cfg=EpropConfig(hidden_neurons=units,extra={'generate_backbone':True})
                config,stats=build_reservoir(seed,condition,cfg)
                save_config(target,config)
                target.with_suffix('.json').write_text(json.dumps(stats,indent=2)+'\n')
            frozen=ROOT/'eval'/'test'/'backbones'/target.parent.name/target.name
            frozen.parent.mkdir(parents=True,exist_ok=True)
            if frozen.exists():
                public,private=load_config(target),load_config(frozen)
                if public.keys()!=private.keys() or any(not np.array_equal(public[k],private[k]) for k in public):
                    raise ValueError(f'public chip differs from frozen evaluator asset: {target}')
            else:
                shutil.copyfile(target,frozen)
            print('fixed backbone',target.relative_to(ROOT),flush=True)
if __name__=='__main__': main()
