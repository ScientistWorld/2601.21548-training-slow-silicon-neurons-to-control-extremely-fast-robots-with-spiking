# Evaluation Targets

All targets are computed by the evaluators from closed-loop rollouts of the submitted controllers. Run `bash scripts/evaluate_train.sh` for the visible slice; the held-out slice is `bash scripts/evaluate_test.sh`.

## Primary: randomized_air_hockey
Reliable interception under randomized launch position and speed.
- **Metric**: `success_rate` (higher is better); also `seed_worst_success` (higher) and `hit_rate` (higher)
- **Reference (held-out)**: 0.9997 success; random policy 0.098

## Primary: velocity_range_wide
Interception across the widest main speed range [0.7, 1.5] m/s.
- **Metric**: `success_rate` (higher is better); also `seed_worst_success`, `hit_rate`
- **Reference (held-out)**: 0.9992 success; random policy 0.101

## Primary: velocity_range_extreme
Interception across [0.5, 2.0] m/s, where a fixed-size network loses accuracy.
- **Metric**: `success_rate` (higher is better); also `seed_worst_success`, `hit_rate`
- **Reference (held-out)**: 0.974 success, worst seed 0.907; random policy 0.098

## Primary: single_core_extreme
Interception across [0.5, 2.0] m/s with a 256-unit controller.
- **Metric**: `success_rate` (higher is better); also `seed_worst_success`, `hit_rate`
- **Reference (held-out)**: 0.931 success, worst seed 0.845; random policy 0.100

## Constraint: controller capacity (all experiments)
Success must be obtained within the experiment's hidden-unit budget: 1020 units, or 256 for `single_core_extreme`.
- **Metric**: `controller_units` (lower is better), counted from the submitted configuration. A controller above the budget fails the whole experiment.

## Constraint: independent, valid controllers (all experiments)
Exactly the 10 files `seed_0.npz` ... `seed_9.npz` must load, validate, and differ from each other. Otherwise the experiment is scored as failed (success 0).
