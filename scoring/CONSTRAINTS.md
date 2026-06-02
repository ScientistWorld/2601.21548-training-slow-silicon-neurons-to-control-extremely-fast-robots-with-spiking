# Scientist Constraints

## Controller Capacity
The hidden spiking reservoir must use no more than 1020 neurons for the main `randomized_air_hockey` score. The paper's claim is about high success with a compact neuromorphic controller; scaling to a much larger network would sidestep the constraint.

## Observation and Action Interface
The agent must use the provided 6D state and two discrete motion primitives. Adding privileged state, continuous low-level torque control, or extra action primitives changes the problem and is not comparable.

## Data and Evaluation
Training and evaluation must use only the procedural task implemented in `data/air_hockey_env.py`, with held-out evaluation seeds. Evaluation artifacts must be scored through `scripts/evaluate.sh`.

## Compliance Checklist
- [ ] Main result uses at most 1020 hidden/reservoir neurons
- [ ] Inputs are limited to puck position, puck velocity, and end-effector position
- [ ] Actions are limited to the two benchmark motion primitives
- [ ] Scores are produced by `scripts/evaluate.sh` from method artifacts
