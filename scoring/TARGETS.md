# Evaluation Targets

## Primary: randomized_air_hockey
Tests whether a controller can reliably complete the central randomized puck interception task.
- **Metric**: `success_rate` (higher is better)
- **Run**: `bash scripts/evaluate.sh` after producing artifacts under `outputs/`

## Constraint: randomized_air_hockey
Tests whether success is obtained within the compact-controller capacity central to the paper's claim.
- **Metric**: `hidden_neurons` (lower is better)
- **Run**: `bash scripts/evaluate.sh` after producing artifacts under `outputs/`

## Primary: variable_speed_generalization
Tests whether a controller remains reliable when puck speed is randomized while launch position is fixed.
- **Metric**: `success_rate` (higher is better)
- **Run**: `bash scripts/evaluate.sh` after producing artifacts under `outputs/`

## Constraint: variable_speed_generalization
Tests whether speed-generalization performance is obtained with the same compact controller size.
- **Metric**: `hidden_neurons` (lower is better)
- **Run**: `bash scripts/evaluate.sh` after producing artifacts under `outputs/`

## Primary: velocity_range_wide
Tests whether the controller remains effective when the puck speed is sampled from the widest range emphasized in the paper.
- **Metric**: `success_rate` (higher is better)
- **Run**: `bash scripts/evaluate.sh` after producing artifacts under `outputs/`

## Constraint: velocity_range_wide
Tests whether wide-range performance is obtained with the same compact controller size.
- **Metric**: `hidden_neurons` (lower is better)
- **Run**: `bash scripts/evaluate.sh` after producing artifacts under `outputs/`
