# Learning-budget audit (final hardening)

`audit.py` runs, resumably: `smoke` (protocol, failure handling and physics
regression at tiny scale), `select` (Eq. 1 learning rate on the public split),
`canonical` (reference and fixed baselines through the normal no-argument
scripts, alone), `audit` (alternative learners on the public split and
variants/shortcuts through the hidden evaluator), `better` (improved learning
rules chosen on the public split, then through the hidden evaluator) and
`analysis` (paired per-chip comparisons, cohort ranks, determinism check).

Results (all evaluator-written):

- `results/smoke.json`, `results/selection.json`, `results/better.json`,
  `results/public_split.json`
- `results/canonical_scores_{train,test}.json`: the reference-only runs that
  became `scoring/scores_{train,test}.json`
- `results/audit_scores_test.json`: every variant and shortcut on the hidden test
- `results/summary.json`: ranks, paired bootstrap intervals over chips,
  reference noise, determinism, train/test comparison
- `results/previous_gym_scores_test.json`: the superseded gym's test scores

Learners in `learners/` are audit controls and shortcut attempts, plus
`eprop_variants.py` (the improved learning rules used for the headroom check).
Run with `python redteam/learning_budget/audit.py [stage ...]` on a compute node.
