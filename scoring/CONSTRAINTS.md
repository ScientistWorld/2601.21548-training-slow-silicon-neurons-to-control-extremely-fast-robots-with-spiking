# Scientist Constraints

## Controller Capacity
Submissions must report the number of adaptive controller units used for each scored result. The main `randomized_air_hockey` reference budget is 1020 units. Scaling to a much larger controller sidesteps the compact-control constraint and should be penalized through `controller_units`.

## Observation and Action Interface
The agent must use the provided 6D state and two discrete motion primitives. Adding privileged state, continuous low-level torque control, or extra action primitives changes the problem and is not comparable.

## Data and Evaluation
Training and evaluation must use only the procedural task implemented in `data/air_hockey_env.py`, with held-out evaluation seeds. Evaluation artifacts must be scored through `scripts/evaluate.sh`.

## Compliance Checklist
- [ ] Main result reports `controller_units` and stays near the 1020-unit reference budget
- [ ] Inputs are limited to puck position, puck velocity, and end-effector position
- [ ] Actions are limited to the two benchmark motion primitives
- [ ] Scores are produced by `scripts/evaluate.sh` from method artifacts
