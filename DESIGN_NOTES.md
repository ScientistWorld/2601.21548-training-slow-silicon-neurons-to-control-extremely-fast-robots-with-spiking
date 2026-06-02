# Train/Test Evaluation Design Notes

## Checked

- Compared `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` for copy-paste mistakes. Train calls `eval/train/evaluate.py` and writes `scoring/scores_train.json`; test calls `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Confirmed the split evaluators are artifact-only: they read JSON files under `outputs/` and do not import or rerun `method/`.
- Grepped for train/test cross-imports and paper-agreement metric names. No cross-imports were present, and the scoring metrics measure task success, contact, compactness, learning progress, and seed stability rather than deviation from published numbers.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Ran `bash -n` on both split wrapper scripts.

## Changed

- Fixed a method-enumeration shortcut in `eval/train/evaluate.py`: the train slice no longer filters default-discovered artifacts to methods already listed in `scoring/reference.json`. Future method variants emitted under `outputs/*.json` are now scored instead of silently ignored.
- Made train scoring emit `hit_rate`, `success_rate_iqr`, and `seed_worst_success` when those fields can be derived from the existing artifact interface. This better matches the declared metric schema and makes the visible slice less one-dimensional.
- Made test scoring emit `seed_worst_success` from `eval_success_by_seed`. This adds a held-out stability angle that penalizes methods tuned to favorable random seeds.
- Completed metric definitions in `scoring/reference.json` by adding required one-sentence descriptions for every metric with nonzero coefficient.
- Added `seed_worst_success` to both existing experiments with a positive coefficient, so a future agent cannot improve the visible mean success alone while hiding brittle per-seed behavior.

## Runtime Concerns For Next Iteration

- The split depends on method artifacts containing seed-level arrays (`train_tail_success_by_seed` for train and `eval_success_by_seed` for test) to compute the strongest stability metric. Current method and baseline artifacts appear to provide `eval_success_by_seed`; the paper method also provides `train_tail_success_by_seed`. Runner iteration should verify those fields are present after a real run.
- The current job script uses `SEEDS=1`, so `success_rate_iqr` and `seed_worst_success` will not be very discriminative until the runner increases seeds. This is acceptable for static design work but should be revisited if compute budget allows.
- `train_success_gain` is naturally unavailable for methods that only emit final evaluation summaries. The evaluator omits missing optional metrics rather than fabricating values; runner iteration should decide whether future method contracts should require training-curve summaries.

## Completed Scoring Run

- Job `dcb87ff5-d90` ran `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` on CPU over the existing `outputs/*.json` artifacts and exited with code 0.
- The generated split scores contain 20 train metric values and 18 test metric values across `randomized_air_hockey` and `velocity_range_wide`.
- Proposed-method primary success differs across slices without collapsing: `randomized_air_hockey` is 0.80 train vs 0.96 test, and `velocity_range_wide` is 0.71 train vs 0.674 test. The random baseline is unchanged across train/test because it has no training-tail summary and is scored from final evaluation artifacts on both slices.
- `success_rate_iqr` saturated at 0.0 for every method because the reproduction artifacts use one seed. It is now retained as a diagnostic field but has coefficient 0.0 in `scoring/reference.json`; `seed_worst_success` remains the weighted seed-level metric available from the current artifact contract.
