"""Summarize evaluator-written results, with explicit rank-normalization limits."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path('/home/user')
BASE = ROOT / 'redteam'

def load(path):
    return json.loads(path.read_text())

def result(scores, experiment):
    return scores.get('experiments', {}).get(experiment, {}).get('results', {}).get('spiking_eprop', {})

def main():
    reference = load(BASE / '_initial/scoring/scores_test.json')
    schema = load(BASE / '_initial/scoring/reference.json')['experiments']
    cases = load(BASE / 'cases.json')
    scores = {c: load(BASE / c / 'scores_test.json') for c in cases if (BASE / c / 'scores_test.json').exists()}
    records = {c: load(BASE / c / 'run_record.json') for c in scores}
    valid = {c: s for c, s in scores.items()
             if records[c]['test_exit_code'] == 0 and c != 'shortcut_nonfinite_values'}
    scales = {}
    for experiment, payload in schema.items():
        scales[experiment] = {}
        for metric, spec in payload['metrics'].items():
            if not abs(spec['coefficient']):
                continue
            values = [result(reference, experiment).get(metric)] + [result(s, experiment).get(metric) for s in valid.values()]
            values = [v for v in values if isinstance(v, (float, int)) and math.isfinite(v)]
            scales[experiment][metric] = statistics.pstdev(values) if len(values) > 1 else 0.0
    def improvement(candidate, target):
        total = 0.0
        detail = {}
        for experiment, payload in schema.items():
            detail[experiment] = {}
            a, b = result(candidate, experiment), result(target, experiment)
            for metric, spec in payload['metrics'].items():
                if not abs(spec['coefficient']):
                    continue
                x, y = a.get(metric), b.get(metric)
                if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
                    continue
                delta = (x - y) * (1 if spec['higher_is_better'] else -1)
                detail[experiment][metric] = delta
                scale = scales[experiment].get(metric, 0)
                if scale:
                    total += payload['weight'] * abs(spec['coefficient']) * delta / scale
        return total, detail
    summary = {'rank_note': 'Illustrative population-SD normalization over finite successful audit attempts plus packaged reference. Future attempt-dependent denominators are unknown; use metric dominance and exact ties for unconditional conclusions.',
               'normalization_scales': scales, 'attempts': {}}
    rows = []
    for case, data in scores.items():
        rank, deltas = improvement(data, reference)
        usable = case in valid
        a = result(data, 'randomized_air_hockey')
        b = result(data, 'velocity_range_wide')
        summary['attempts'][case] = {'test_exit_code': records[case]['test_exit_code'],
            'illustrative_rank_delta': rank if usable else None,
            'directed_deltas': deltas,
            'results': {e: result(data, e) for e in schema}}
        rows.append([case, records[case]['test_exit_code'], a.get('success_rate'), b.get('success_rate'),
                     a.get('controller_units'), b.get('controller_units'),
                     a.get('seed_worst_success'), b.get('seed_worst_success'), rank if usable else None])
    seed_summary = {}
    for experiment in schema:
        per_seed = [result(scores[f'quality_reference_seed{s}'], experiment)['success_rate'] for s in range(3)]
        seed_summary[experiment] = {'values': per_seed, 'mean': statistics.mean(per_seed),
                                   'min': min(per_seed), 'max': max(per_seed),
                                   'max_minus_min': max(per_seed) - min(per_seed)}
    per_seed_ranks = [improvement(scores[f'quality_reference_seed{s}'], reference)[0] for s in range(3)]
    summary['reference_seed_spread'] = seed_summary
    summary['reference_rank_spread'] = {'values': per_seed_ranks, 'max_minus_min': max(per_seed_ranks) - min(per_seed_ranks)}
    summary['quality_comparison'] = {}
    for case in ('quality_weak_no_learning', 'quality_more_training', 'quality_learned_timing'):
        rank, deltas = improvement(scores[case], scores['quality_reference_three_seeds'])
        summary['quality_comparison'][case] = {'illustrative_rank_change_vs_matched_reference': rank,
                                              'directed_deltas_vs_matched_reference': deltas}
    (BASE / 'analysis.json').write_text(json.dumps(summary, indent=2, allow_nan=True) + '\n')
    with (BASE / 'summary.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'test_exit_code', 'random_success', 'wide_success', 'random_units', 'wide_units',
                         'random_worst_seed', 'wide_worst_seed', 'illustrative_rank_delta'])
        writer.writerows(rows)
    print(json.dumps({'seed_spread': seed_summary, 'seed_rank_spread': summary['reference_rank_spread'],
                      'quality_comparison': summary['quality_comparison']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
