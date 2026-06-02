# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

- `success_rate`: fraction of evaluation episodes where the puck crosses `x_p > 1.5 m` with forward velocity `v_x > 0.1 m/s`. Higher is better.
- `hit_rate`: fraction of episodes where the mallet contacts the puck within the 0.08 m contact threshold. Higher is better.
- `hidden_neurons`: hidden reservoir size used by the controller. Lower is better as a capacity constraint.

## Baselines and Targets

The main target is the randomized initial position and speed setting. The paper reports stabilization above 98% success after roughly 1300 training episodes using a 1020-neuron reservoir. A random-action policy is included to verify that the evaluation task is non-trivial.

## Evaluation Protocol

Train on the randomized-position-and-speed condition and evaluate on held-out random seeds from the same condition. Report the mean success and hit rates over all evaluation episodes. Evaluation code must consume method artifacts written under `outputs/` and must not import from `method/`.
