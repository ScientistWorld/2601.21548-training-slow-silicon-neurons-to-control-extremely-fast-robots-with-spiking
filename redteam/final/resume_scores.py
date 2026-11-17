"""Resume this audit from raw rollouts, without changing normal evaluation.

Worker drains restart the whole job. Audit checkpoints live in the split's
private raw directory and are keyed by exact inputs, controller arrays and
evaluator sources. No metric, status or submitted cache is reused. The public
evaluate_{train,test}.sh entry points do not import this module or use caches.
"""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def source_files(evaluator):
    if evaluator.__name__.startswith('eval.test.'):
        return [ROOT / name for name in (
            'eval/test/evaluate.py', 'eval/test/task_sim.py',
            'eval/test/chip_runtime.py', 'eval/test/robot/air_hockey.xml')]
    return [ROOT / name for name in (
        'eval/train/evaluate.py', 'data/air_hockey_env.py',
        'data/neuromorphic_chip.py', 'data/robot/air_hockey.xml')]


def array_digest(digest, value):
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())


class RawResume:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        digest = hashlib.sha256(b'raw-audit-checkpoint-v1')
        for path in [*source_files(evaluator), Path(__file__)]:
            digest.update(path.read_bytes())
        import mujoco
        digest.update((np.__version__ + mujoco.__version__).encode())
        self.source_digest = digest.digest()
        self.directory = evaluator.RAW / 'audit_resume'
        self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, kind, init, config=None, arguments=()):
        digest = hashlib.sha256(self.source_digest)
        digest.update(kind.encode())
        digest.update(json.dumps(arguments).encode())
        array_digest(digest, init)
        if config is not None:
            for name, value in sorted(config.items()):
                digest.update(name.encode())
                array_digest(digest, value)
        return digest.hexdigest()

    @staticmethod
    def validate(result, count):
        if len(result) != 2:
            raise ValueError('raw rollout must contain success and contact')
        for values in result:
            if np.shape(values) != (count,) or not np.all(np.isin(values, [0, 1])):
                raise ValueError('invalid raw rollout')

    def store(self, key, result, count):
        self.validate(result, count)
        target = self.directory / f'{key}.npz'
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, success=result[0], hit=result[1])
        temporary.replace(target)

    def obtain(self, key, count, compute):
        target = self.directory / f'{key}.npz'
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                result = saved['success'].copy(), saved['hit'].copy()
            self.validate(result, count)
            print(f'audit raw resume {key[:12]}', flush=True)
            return result
        result = compute()
        self.store(key, result, count)
        print(f'audit raw completed {key[:12]}', flush=True)
        return result

    def adopt_cancelled_run(self):
        """Import the nine completed rollouts from the unchanged final setup.

        These were made by our cancelled, stable audit, before fingerprints
        were introduced. Refuse adoption if any evaluator source differs from
        that run's commit. This one-off migration is never used by the gym.
        """
        evaluator = self.evaluator
        marker = self.directory / 'legacy_adopted.json'
        if marker.exists() or not evaluator.__name__.startswith('eval.test.'):
            return
        for path in source_files(evaluator):
            previous = subprocess.check_output(
                ['git', 'show', '8b99f32:' + str(path.relative_to(ROOT))], cwd=ROOT)
            if previous != path.read_bytes():
                raise RuntimeError('cancelled-run evaluator changed; cannot adopt old raw outcomes')
        imported = []
        for method in evaluator.discover_methods():
            tag = hashlib.sha256(method.encode()).hexdigest()[:24]
            for experiment in evaluator.EXPERIMENTS:
                path = evaluator.RAW / f'{tag}_{experiment}.npz'
                if not path.exists():
                    continue
                configs = evaluator.load_controllers(evaluator.OUTPUTS / method, experiment)
                init = evaluator.held_out_episodes(experiment)
                with np.load(path, allow_pickle=False) as saved:
                    outcome = {name: saved[name].copy() for name in saved.files}
                evaluator.summarize(outcome)
                units = np.array([evaluator.controller_units(c) for c in configs])
                if not np.array_equal(outcome['units'], units):
                    raise ValueError('legacy raw unit count differs from actual controllers')
                for k, config in enumerate(configs):
                    self.store(self.key('controller', init, config),
                               (outcome['success'][k], outcome['hit'][k]), len(init))
                imported.append(str(path.relative_to(evaluator.RAW)))
        marker.write_text(json.dumps({'origin_job': 'be37540d-3d6', 'raw_files': imported}, indent=2) + '\n')
        print(f'adopted {len(imported)} complete raw experiments from cancelled stable audit', flush=True)

    def install(self):
        evaluator = self.evaluator
        controller = evaluator.rollout_controller
        baseline = evaluator.rollout_baseline
        score_task = evaluator.score_task
        invalid = {}
        # The normal evaluator ultimately fails every experiment if any is
        # invalid. Preflight the same constraints to avoid simulating valid
        # pieces of a submission that is already certain to fail overall.
        for method in evaluator.discover_methods():
            try:
                for experiment in evaluator.EXPERIMENTS:
                    evaluator.load_controllers(evaluator.OUTPUTS / method, experiment)
            except Exception as exc:
                invalid[method] = f'{type(exc).__name__}: {exc}'

        def run_controller(config, init):
            return self.obtain(self.key('controller', init, config), len(init),
                               lambda: controller(config, init))

        def run_baseline(name, init, stream, experiment, lead=.15):
            key = self.key('baseline', init, arguments=(name, stream, experiment, lead))
            return self.obtain(key, len(init), lambda: baseline(name, init, stream, experiment, lead))

        def run_task(task):
            method, experiment, is_baseline = task
            if not is_baseline and method in invalid:
                return method, experiment, evaluator.worst(experiment), {
                    'status': 'failed', 'reason': invalid[method]}
            return score_task(task)

        evaluator.rollout_controller = run_controller
        evaluator.rollout_baseline = run_baseline
        evaluator.score_task = run_task


def main(split):
    if split not in ('train', 'test'):
        raise ValueError('unknown split')
    evaluator = importlib.import_module(f'eval.{split}.evaluate')
    resume = RawResume(evaluator)
    resume.adopt_cancelled_run()
    resume.install()
    return evaluator.main()


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
