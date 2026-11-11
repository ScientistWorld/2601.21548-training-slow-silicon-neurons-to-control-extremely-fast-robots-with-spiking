"""Degenerate and enumeration shortcuts, using only the visible simulator."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np

from data.air_hockey_env import AirHockeyInterceptionEnv

CONDITIONS = ('random_position_speed', 'speed_wide')


def key(state):
    return tuple(float(x) for x in state)


def best_delay(env):
    """Enumerate 31 candidate waits from the visible initial observation.

    Clone an environment reset from its public state, without reading future
    outcomes or evaluator code. Dynamics are deterministic after reset.
    """
    for delay in range(31):
        trial = AirHockeyInterceptionEnv(env.condition)
        trial.state[:] = env.state
        done = False
        step = 0
        while not done:
            _, _, done, info = trial.step(int(step >= delay))
            step += 1
        if info['success']:
            return delay
    return 0


def build_lookup(condition):
    """Use seeds disclosed by visible method/train.py; no hidden input."""
    lookup = {}
    for seed in range(3):
        env = AirHockeyInterceptionEnv(condition, seed=9000 + seed)
        for _ in range(500):
            state = env.reset()
            lookup[key(state)] = best_delay(env)
    return lookup


def rollouts(policy, output_folder, seeds=range(3), count=500, units=1020):
    artifacts = []
    for condition in CONDITIONS:
        lookup = build_lookup(condition) if policy == 'seed_lookup' else None
        successes, hits = [], []
        details = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            env = AirHockeyInterceptionEnv(condition, seed=9000 + seed)
            records = []
            for _ in range(count):
                state = env.reset()
                delay = best_delay(env) if policy == 'bruteforce' else None
                delay = lookup.get(key(state)) if lookup is not None else delay
                step = 0
                done = False
                while not done:
                    if policy == 'home':
                        action = 0
                    elif policy == 'random':
                        action = int(rng.integers(2))
                    elif policy in ('bruteforce', 'seed_lookup'):
                        action = int(delay is not None and step >= delay)
                    else:
                        action = 1
                    _, _, done, info = env.step(action)
                    step += 1
                records.append({'success': bool(info['success']), 'hit': bool(info['hit']),
                                'steps': step, 'selected_delay': delay})
            successes.append(float(np.mean([r['success'] for r in records])))
            hits.append(float(np.mean([r['hit'] for r in records])))
            details.append({'seed': seed, 'records': records})
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        (Path(output_folder) / f'rollouts_{condition}.json').write_text(json.dumps(details) + '\n')
        artifacts.append({
            'method': 'spiking_eprop', 'condition': condition,
            'controller_units': units, 'episodes': 0, 'eval_episodes': count,
            'seeds': len(successes),
            'success_rate': float(np.mean(successes)), 'hit_rate': float(np.mean(hits)),
            'eval_success_by_seed': successes,
            'success_rate_iqr': float(np.subtract(*np.percentile(successes, [75, 25]))),
        })
    return artifacts
