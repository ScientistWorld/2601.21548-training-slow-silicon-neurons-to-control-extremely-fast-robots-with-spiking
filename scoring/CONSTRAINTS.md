# Scientist Constraints

## Controller Capacity
- Each experiment has a hidden-unit budget: 1020 units, or 256 for `single_core_extreme`.
- The evaluator counts units from the submitted configuration. A controller above budget fails the experiment.
- Fewer units within the budget score slightly better (`controller_units`).

## Controller Substrate and Interface
- Controllers must be expressed in the configuration format executed by `data/neuromorphic_chip.py`. That format covers population-coded spike generators, leaky integrate-and-fire hidden units with bounded parameters, optional recurrence, and a two-score linear readout on filtered spikes.
- At every 20 ms control step the controller sees only the 6D state (puck position and velocity, end-effector position) and chooses one of the two motion primitives.
- No other computation happens at decision time. Lookups, simulators or planners outside the network cannot run inside the evaluator.

## Training
- Learning must come from interaction with the public simulator `data/air_hockey_env.py`. Any training procedure is allowed, but the paper's setting is online learning from a modest number of episodes (2000 per run).
- Supervising the network with per-episode plans computed by simulating candidate strike times is out of scope. So is hand-writing the intercept rule into the weights.
- Each of the 10 controllers per experiment must come from an independent training run. Training more runs and submitting only the best ones misrepresents run-to-run reliability.

## Data and Evaluation
- Held-out episodes, the frozen simulator and runtime copies, the baselines, the episode count and the budgets are fixed by `eval/`.
- Methods write controllers under `outputs/<method>/<experiment>/`. Numbers written by a method are never read.

## Compliance Checklist
- [ ] Ten controllers per experiment, each from its own training run (no selection among extra runs)
- [ ] Every controller within the experiment's unit budget
- [ ] Controllers learned from interaction, not distilled from simulated per-episode plans or a hand-derived intercept rule
- [ ] Inputs limited to the 6D benchmark state; actions limited to the two primitives
- [ ] Scores produced by the repository evaluators from the submitted controllers
