# Evaluation

## What Is Submitted

A submission is a set of **controller configurations**, not reported numbers. For each experiment, a method writes 10 independently trained controllers:

```
outputs/<method>/<experiment>/seed_<k>.npz      k = 0..9
```

Each file fully specifies a spiking network in the format executed by the public runtime `data/neuromorphic_chip.py`:
- **Input stage:** per state variable, a population of Gaussian-tuned deterministic spike generators; centers, widths and peak rates are free (at most 32 cells per variable, at most 1000 Hz).
- **Hidden stage:** leaky integrate-and-fire units with synaptic filtering, adaptation and refractoriness (per-unit parameters within physical bounds), input weights, and optional recurrent weights.
- **Readout:** two linear scores on the filtered hidden spike trains.

The evaluator runs 20 ms of network dynamics per control step and takes the action with the higher score. How the configuration was obtained (learning rule, number of training episodes, encoding, connectivity) is entirely up to the method. Training interacts with the public simulator in `data/air_hockey_env.py`.

## Experiments and Metrics

| Experiment | Setting | Unit budget | Weight |
|---|---|---|---|
| `randomized_air_hockey` | launch x 0.95-1.05 m, speed 1.0-1.5 m/s | 1020 | 0.4 |
| `velocity_range_wide` | speed 0.7-1.5 m/s | 1020 | 0.2 |
| `velocity_range_extreme` | speed 0.5-2.0 m/s | 1020 | 0.2 |
| `single_core_extreme` | speed 0.5-2.0 m/s | 256 | 0.2 |

Per experiment the evaluator runs every controller in closed loop on 1000 episodes and computes:
- `success_rate` (coefficient 1.0, higher is better): success pooled over the 10 controllers. An episode succeeds when the puck crosses x > 1.5 m with v_x > 0.1 m/s.
- `seed_worst_success` (0.3, higher): success of the weakest of the 10 controllers.
- `hit_rate` (0.1, higher): fraction of episodes with puck-mallet contact.
- `controller_units` (0.1, lower): hidden units of the largest controller, counted from the configuration.

## Protocol and Failure Rules

- `scripts/evaluate_train.sh` and `scripts/evaluate.sh` score on public validation episodes. `scripts/evaluate_test.sh` (no arguments) scores on held-out episodes with frozen copies of the simulator and runtime. Editing `data/` therefore changes nothing on the held-out test.
- Every method directory under `outputs/` is scored, plus the proposed method listed in `scoring/reference.json`.
- A (method, experiment) entry receives the worst value of every metric (success 0, units = budget) when any of these hold:
  - a controller is missing, unreadable or malformed
  - a value is non-finite or out of bounds
  - the controller exceeds the experiment's unit budget
  - two seeds' controllers are identical
- Baselines are computed by the evaluator on the same episodes: `random_policy` (uniform random primitive every step) and `always_forward` (strike immediately). Both score about 0.10 and 0.00 success.
- Scoring is deterministic: the same files always receive the same scores.
