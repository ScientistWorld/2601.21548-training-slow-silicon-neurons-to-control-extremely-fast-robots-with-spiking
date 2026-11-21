# Targets

| Experiment | Scientific target | Weight |
|---|---|---:|
| randomized_air_hockey | high success reached within 2000 online trials, acquired quickly, randomized position and speed | 0.4 |
| velocity_range_wide | the same with launch speeds 0.7–1.5 m/s | 0.3 |
| velocity_range_extreme | the same with launch speeds 0.5–2.0 m/s | 0.3 |

Benefit metrics (each experiment, higher is better, coefficient 1):
`final_success` (primary; last 500 of 2000 online episodes) and
`online_success` (all 2000 online episodes).

Constraints the paper's claim depends on, enforced by the evaluator rather than
scored: the fixed 1020-unit random substrate (not trainable, not visible), a
linear two-action readout of its filtered spike trains, and exactly 2000
sequential episodes per instance with no state, launch or chip information.
These cannot be traded for a better score: the evaluator owns the substrate,
the episode count and every outcome, and any malformed or failed submission
scores 0 in every metric of every experiment.

Local reference results (Eq. 1, hidden test) and the fixed baselines are in
`reference.json`. The paper's own numbers stay under `_paper_reported`: they
come from silicon and the authors' robot, so they are provenance, not targets.
