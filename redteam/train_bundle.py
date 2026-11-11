"""Real, compute-node-only quality controls. No evaluator code is imported.

Original dynamics and updates are used for the paper-method controls. Evaluation
clones preserve training RNG state when taking intermediate measurements.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time

import numpy as np

from data.air_hockey_env import AirHockeyInterceptionEnv
from method.spiking_eprop_rl import SpikingAgentConfig, SpikingEpropAgent
from method.train import run_episode

ROOT = Path('/home/user')
CACHE = ROOT / 'redteam/_cache'
CONDITIONS = ('random_position_speed', 'speed_wide')


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def evaluate(agent, condition, seed, episodes=500):
    agent = copy.deepcopy(agent)
    env = AirHockeyInterceptionEnv(condition, seed=9000 + seed)
    records = []
    action_counts = [0, 0]
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action, _, _ = agent.act(state, greedy=True)
            action_counts[action] += 1
            state, _, done, info = env.step(action)
        records.append(info)
    return {
        'success': float(np.mean([r['success'] for r in records])),
        'hit': float(np.mean([r['hit'] for r in records])),
        'action_counts': action_counts,
    }


def ensure_trained():
    CACHE.mkdir(parents=True, exist_ok=True)
    # One uninterrupted trajectory for each seed; the longer-training variant
    # therefore shares exactly its first reference-budget interactions.
    for condition in CONDITIONS:
        reference_episodes = 2000 if condition == CONDITIONS[0] else 5000
        for seed in range(3):
            paths = {n: CACHE / f'{condition}_seed{seed}_episodes{n}.json'
                     for n in (0, reference_episodes, 10000)}
            if all(p.exists() for p in paths.values()):
                continue
            start = time.monotonic()
            cfg = SpikingAgentConfig(hidden_neurons=1020,
                learning_rate=0.004 if condition == CONDITIONS[0] else 0.006,
                entropy_floor=0.08 if condition == CONDITIONS[0] else 0.06)
            agent = SpikingEpropAgent(seed=seed, config=cfg)
            env = AirHockeyInterceptionEnv(condition, seed=1000 + seed)
            train = []
            for n in range(10001):
                if n in paths:
                    result = evaluate(agent, condition, seed)
                    first = float(np.mean([r['success'] for r in train[:100]])) if train else 0.0
                    tail = float(np.mean([r['success'] for r in train[-100:]])) if train else 0.0
                    payload = {
                        'method': 'spiking_eprop', 'condition': condition,
                        'seed': seed, 'episodes': n, 'eval_episodes': 500,
                        'seeds': 1, 'controller_units': 1020,
                        'learning_rate': cfg.learning_rate,
                        'entropy_floor': cfg.entropy_floor,
                        'success_rate': result['success'], 'hit_rate': result['hit'],
                        'eval_success_by_seed': [result['success']],
                        'success_rate_iqr': 0.0,
                        'train_first_success_rate': first,
                        'train_tail_success_rate': tail,
                        'train_tail_success_by_seed': [tail],
                        'audit_action_counts': result['action_counts'],
                        'audit_elapsed_seconds': time.monotonic() - start,
                    }
                    write(paths[n], payload)
                    print(f'TRAIN {condition} seed={seed} episodes={n} '
                          f'eval={result["success"]:.4f} actions={result["action_counts"]}', flush=True)
                if n < 10000:
                    train.append(run_episode(env, agent, train=True))


def cached_artifacts(kind, seed=None):
    ensure_trained()
    result = []
    for condition in CONDITIONS:
        n = {'weak': 0, 'more': 10000,
             'reference': 2000 if condition == CONDITIONS[0] else 5000}[kind]
        seeds = range(3) if seed is None else [seed]
        values = [json.loads((CACHE / f'{condition}_seed{s}_episodes{n}.json').read_text())
                  for s in seeds]
        artifact = dict(values[0])
        artifact['seeds'] = len(values)
        artifact['eval_success_by_seed'] = [v['success_rate'] for v in values]
        artifact['train_tail_success_by_seed'] = [v['train_tail_success_rate'] for v in values]
        for key in ('success_rate', 'hit_rate', 'train_tail_success_rate', 'train_first_success_rate'):
            artifact[key] = float(np.mean([v[key] for v in values]))
        artifact['success_rate_iqr'] = float(np.subtract(*np.percentile(artifact['eval_success_by_seed'], [75, 25])))
        artifact['audit_action_counts_by_seed'] = [v['audit_action_counts'] for v in values]
        result.append(artifact)
    return result


class LearnedTimingController:
    """Online episodic RL over delays, with no model, oracle or test-seed input.

    The sole representation choice is to group initial observed velocities and
    learn returns for discrete waiting times. No intercept-time formula is used.
    Learning trials interact with the unmodified environment. This is an honest
    alternative for the toy task, not a reproduction of the spiking algorithm.
    """
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.values = np.zeros((8, 21))
        self.counts = np.zeros((8, 21))

    def context(self, state):
        return int(np.clip((abs(float(state[2])) - 0.7) / 0.1, 0, 7))

    def episode(self, env, train, episode=0):
        state = env.reset()
        context = self.context(state)
        unseen = np.flatnonzero(self.counts[context] == 0)
        if train and len(unseen):
            delay = int(self.rng.choice(unseen))
        elif train and self.rng.random() < max(0.05, 0.5 * (1 - episode / 2000)):
            delay = int(self.rng.integers(21))
        else:
            delay = int(np.argmax(self.values[context]))
        reward_sum = 0.0
        done = False
        step = 0
        while not done:
            state, reward, done, info = env.step(int(step >= delay))
            reward_sum += reward
            step += 1
        if train:
            self.counts[context, delay] += 1
            self.values[context, delay] += (reward_sum - self.values[context, delay]) / self.counts[context, delay]
        return info


def learned_timing():
    artifacts = []
    for condition in CONDITIONS:
        successes, hits, firsts, tails = [], [], [], []
        tables = []
        for seed in range(3):
            agent = LearnedTimingController(seed)
            env = AirHockeyInterceptionEnv(condition, seed=1000 + seed)
            records = [agent.episode(env, True, n) for n in range(2000)]
            firsts.append(float(np.mean([r['success'] for r in records[:100]])))
            tails.append(float(np.mean([r['success'] for r in records[-100:]])))
            test_env = AirHockeyInterceptionEnv(condition, seed=9000 + seed)
            records = [agent.episode(test_env, False) for _ in range(500)]
            successes.append(float(np.mean([r['success'] for r in records])))
            hits.append(float(np.mean([r['hit'] for r in records])))
            tables.append({'seed': seed, 'values': agent.values.tolist(), 'counts': agent.counts.tolist()})
        write(ROOT / 'redteam/quality_learned_timing' / f'learned_tables_{condition}.json', tables)
        artifacts.append({
            'method': 'spiking_eprop', 'condition': condition,
            'episodes': 2000, 'eval_episodes': 500, 'seeds': 3,
            # Hold the declared budget constant to isolate the task benefit.
            # Actual table has 168 values and 168 counts, below this budget.
            'controller_units': 1020,
            'success_rate': float(np.mean(successes)), 'hit_rate': float(np.mean(hits)),
            'eval_success_by_seed': successes,
            'success_rate_iqr': float(np.subtract(*np.percentile(successes, [75, 25]))),
            'train_first_success_rate': float(np.mean(firsts)),
            'train_tail_success_rate': float(np.mean(tails)),
            'train_tail_success_by_seed': tails,
        })
    return artifacts


if __name__ == '__main__':
    ensure_trained()
