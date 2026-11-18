"""Complete the combined audit using the already-trained 64000-trial reference.

The 32000-trial audit is preserved: its strongest simple comparator narrowly
outranked it. No training or physics is changed here. Public-only calibration
tests headroom, then raw paired outcomes certify it and regenerate canonical
reference/baseline scores in isolation.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from redteam.final.study import EXPS,ROOT,STUDY,attacks,preflight,summarize_audit,write_json

def copy_tree(source,target):
    if target.exists(): shutil.rmtree(target)
    shutil.copytree(source,target)

def setup():
    marker=STUDY/'ref64_setup_complete.json'
    if marker.exists(): return
    archive=STUDY/'reference32_audit'
    archive.mkdir(exist_ok=True)
    for name in ['all_scores_test.json','paired_comparisons.json']:
        source=STUDY/name
        target=archive/name
        if source.exists() and not target.exists(): shutil.copyfile(source,target)
    original=ROOT/'tmp'/'reference_32000'
    if not original.exists():
        # Preserve the old reference completely before replacing any file.
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            temporary=Path(directory)/'reference_32000'
            shutil.copytree(ROOT/'outputs/spiking_eprop',temporary)
            os.replace(temporary,original)
    previous=ROOT/'tmp/final_variants'
    copy_tree(previous/'long_training',ROOT/'outputs/spiking_eprop')
    copy_tree(original,ROOT/'outputs/short_training_32000')
    for name in ['averaged_readout','weak_2000','untrained','rule_distillation']:
        copy_tree(previous/name,ROOT/'outputs'/name)
    # Only the identity in the diagnostic history changes; these exact Eq. 1
    # trajectories were already executed to trial 64000 in the combined job.
    for e in EXPS:
        for seed in range(10):
            path=ROOT/'outputs/spiking_eprop'/e/f'training_seed_{seed}.json'
            history=json.loads(path.read_text())
            if history['episodes']!=64000: raise ValueError('wrong reference checkpoint')
            history['method']='spiking_eprop'
            write_json(path,history)
    attacks()
    write_json(marker,{'reference_trials':64000,'source':'previous exact Eq. 1 trial-64000 snapshots',
                       'training_unchanged':True,'physics_unchanged':True})

def checks(scores,reference,bootstrap):
    from eval.rank import rank
    ranks=rank(scores,reference)
    report={'ranks':ranks,'paired_bootstrap':bootstrap,
            'primary_headroom_variant':'public_calibrated',
            'honest_gain_exceeds_paired_noise':bootstrap['public_calibrated']['paired_bootstrap_95_lower']>0,
            'timing_rules_below_reference':all(ranks[m]<0 for m in ['intercept_rule','rule_distillation']),
            'weak_and_simple_below_reference':all(ranks[m]<0 for m in
                ['short_training_32000','weak_2000','untrained','random_policy','always_forward']),
            'reference_at_bound':{e:spec['results']['spiking_eprop']['success_rate']==1.
                                  for e,spec in reference['experiments'].items()},
            'honest_variant_at_all_bounds':{m:all(scores['experiments'][e]['results'][m]['success_rate']==1.
                                           for e in EXPS) for m in bootstrap}}
    report['passed']=(report['honest_gain_exceeds_paired_noise'] and report['timing_rules_below_reference']
             and report['weak_and_simple_below_reference'] and not any(report['honest_variant_at_all_bounds'].values()))
    write_json(STUDY/'scientific_checks.json',report)
    write_json(STUDY/'ranks.json',ranks)
    print('SCIENTIFIC CHECKS',report,flush=True)
    return report['passed']

def main():
    os.chdir(ROOT)
    preflight()
    setup()
    marker=STUDY/'ref64_audit_complete.json'
    if not marker.exists():
        subprocess.run([sys.executable,'method/calibrate_readout.py',
             '--average-root',str(ROOT/'tmp/final_variants/averaged_readout')],check=True)
        shutil.copyfile(ROOT/'outputs/public_calibrated/selection.json',STUDY/'public_calibration_selection.json')
        subprocess.run([sys.executable,'redteam/final/resume_scores.py','test'],check=True)
        scores=json.loads((ROOT/'scoring/scores_test.json').read_text())
        write_json(STUDY/'all_scores_ref64.json',scores)
        summarize_audit(scores)
        from eval.test.paired_audit import compare
        reference=json.loads((ROOT/'scoring/reference.json').read_text())
        bootstrap=compare(scores,reference,['public_calibrated','sharpened_readout','averaged_readout'])
        write_json(STUDY/'paired_bootstrap_ref64.json',bootstrap)
        write_json(marker,{'complete':True})
    scores=json.loads((STUDY/'all_scores_ref64.json').read_text())
    bootstrap=json.loads((STUDY/'paired_bootstrap_ref64.json').read_text())
    # No training, calibration or audit worker remains during canonical scoring.
    archive=ROOT/'tmp/ref64_variants'
    archive.mkdir(exist_ok=True)
    for path in (ROOT/'outputs').iterdir():
        if path.is_dir() and path.name!='spiking_eprop':
            destination=archive/path.name
            if destination.exists(): shutil.rmtree(destination)
            shutil.move(str(path),str(destination))
    for split in ['train','test']:
        subprocess.run([sys.executable,'redteam/final/resume_scores.py',split],check=True)
    shutil.copyfile(ROOT/'scoring/scores_train.json',ROOT/'scoring/scores.json')
    reference=json.loads((ROOT/'scoring/reference.json').read_text())
    final=json.loads((ROOT/'scoring/scores_test.json').read_text())
    for e,spec in reference['experiments'].items():
        spec['results']={m:{'type':'proposed' if m=='spiking_eprop' else 'baseline',**values}
                        for m,values in final['experiments'][e]['results'].items()}
    reference['_setup_note']=reference['_setup_note'].replace(
        'Local reference results are populated after the combined audit; zero placeholders are not measured results.',
        'Local reference results below were recomputed from isolated rollouts after all training and calibration completed.')
    write_json(ROOT/'scoring/reference.json',reference)
    passed=checks(scores,reference,bootstrap)
    subprocess.run([sys.executable,'validate.py'],check=True)
    if not passed: raise RuntimeError('scientific hardening is not certified; inspect scientific_checks.json')
    print('FINAL REFERENCE-64 AUDIT COMPLETE',flush=True)

if __name__=='__main__': main()
