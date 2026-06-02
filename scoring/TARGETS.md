# Evaluation Targets

## Primary: randomized_air_hockey
Tests whether a controller can reliably complete the central randomized puck interception task.
- **Metric**: `success_rate` (higher is better)
- **Run**: `bash scripts/evaluate.sh` after producing artifacts under `outputs/`

## Constraint: randomized_air_hockey
Tests whether success is obtained within the compact-controller capacity central to the paper's claim.
- **Metric**: `hidden_neurons` (lower is better)
- **Run**: `bash scripts/evaluate.sh` after producing artifacts under `outputs/`
