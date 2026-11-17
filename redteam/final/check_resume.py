"""Cheap audit-only check of raw checkpoint reuse and invalidation."""
import tempfile
from pathlib import Path
import numpy as np
from eval.test import evaluate
from redteam.final.resume_scores import RawResume


def main():
    original = evaluate.RAW
    original.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=original) as temporary:
        evaluate.RAW = Path(temporary)
        try:
            cache = RawResume(evaluate)
            initial = np.array([[1., .5], [1., 2.]])
            config = {'w_out': np.array([[0., 1.]], dtype=np.float32)}
            key = cache.key('controller', initial, config)
            calls = []

            def compute():
                calls.append(1)
                return np.array([True, False]), np.array([True, True])

            first = cache.obtain(key, 2, compute)
            second = cache.obtain(key, 2, compute)
            assert len(calls) == 1
            assert all(np.array_equal(a, b) for a, b in zip(first, second))
            different_config = {'w_out': config['w_out'] + .01}
            assert cache.key('controller', initial, different_config) != key
            assert cache.key('controller', initial + .01, config) != key
            assert cache.key('baseline', initial, arguments=('a', 0)) != key
            cache.source_digest = b'changed evaluator'
            assert cache.key('controller', initial, config) != key
            try:
                cache.store('invalid', (np.array([2, 0]), np.array([1, 1])), 2)
            except ValueError:
                pass
            else:
                raise AssertionError('impossible raw outcome accepted')
            assert not (cache.directory / 'invalid.npz').exists()
        finally:
            evaluate.RAW = original
    print('audit raw resume and invalidation checks passed', flush=True)


if __name__ == '__main__':
    main()
