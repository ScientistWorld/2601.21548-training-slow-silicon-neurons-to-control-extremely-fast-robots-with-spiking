# Evaluation

Write `outputs/<method>/<experiment>/seed_<k>.npz` for k=0..9. Each slot is
assigned the chip in `data/backbones/<condition>_<units>/seed_<k>.npz`.
Supply its full configuration with your own `w_out` and `b_out` arrays. Every
other array must match the assigned instance exactly. The public runtime in
`data/neuromorphic_chip.py` defines the file format and bounded values.
The evaluator executes this configuration and samples the softmax action probabilities,
using identical uniform draws for every method at each episode and tick.
Any learning procedure may produce the readout. Evaluation never trains.

| Experiment | Condition | Units | Weight |
|---|---|---:|---:|
| `velocity_range_extreme` | lateral speeds 0.5–2.0 m/s | 1020 | 0.2 |
| `single_core_extreme` | same speeds | 256 | 0.3 |
| `sensor_robustness` | varied positions/headings, state noise and latency | 1020 | 0.5 |

Each of ten controllers sees the same 4096 episodes. All methods see the same
inputs, including the same observation-noise and action-sampling draws. Position noise has standard
deviation 0.035 m and velocity noise 0.35 m/s in the robustness setting. Its
observations arrive two control ticks (40 ms) late. Train and test use independently
drawn episodes of identical difficulty and separate evaluator implementations.

`success_rate` is the scored metric (higher is better; coefficient 1).
`seed_worst_success`, `hit_rate`, and `controller_units` are diagnostics
(coefficient 0). Capacity is a hard constraint, not a compensating reward.
Ranking sums experiment weight times absolute metric coefficient times the
better-direction improvement over the local reference, divided by the metric's
standard deviation over the attempts being compared. A constant metric
contributes zero. `eval/rank.py` provides this calculation for audit cohorts.

Run `bash scripts/evaluate_train.sh` for `scoring/scores_train.json` and
`bash scripts/evaluate_test.sh` with no arguments for `scoring/scores_test.json`.
Every directory under `outputs/` is scored, including hidden names, as is the
reference even if its files are missing. Fixed evaluator baselines have reserved
names `random_policy`, `always_forward`, and `intercept_rule`.
Only numbers under each experiment's `results` count. Reported metrics, logs,
flags and other files supplied by a method are ignored.

Any missing experiment or controller, invalid value, wrong chip, or capacity
violation fails the entire submission: every experiment gets success, worst
success and contact rate zero, and controller units equal to its budget.
Nonfinite, reset or stalled physics also fails the run. Numerical safety
bounds are 50 m/s puck speed, 100 rad/s joint speed, and 10 cm penetration
beyond a table boundary; these are checked before scoring any goal.
The test uses its own frozen assets. Only bounded numeric configurations cross
its interface, so a method cannot query hidden trajectories with a private
simulator during evaluation. Baselines and their candidate operating points are fixed by
the evaluator. The intercept baseline reports the best global rule in a fixed
27-rule bank with three observation smoothing constants (0, 40, 80 ms), giving the simple comparison a conservative advantage. There are no wall-clock scoring metrics or method time limits.
