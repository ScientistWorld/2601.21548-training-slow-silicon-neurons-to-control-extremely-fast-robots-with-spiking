# Constraints

A submission is a learning procedure (`outputs/<method>/learner.py`, see
`briefing/evaluation.md`). The evaluator, not the method, decides:

- the controller instances (fixed random spiking substrates, 1020 units; the
  hidden test's 20 instances are not available anywhere else);
- the launches, their order, the action-sampling random numbers and the
  2000-episode budget per instance;
- every outcome, computed from its own simulation of the robot.

The policy is a linear readout `(w [1020, 2], b [2])` of the substrate's filtered
spike trains, fixed within an episode; actions are sampled from its softmax.
Between episodes the learner may compute anything from what it has been given.
Each instance's learner runs in its own process and must not communicate with
the others, read files under `eval/`, or identify or reconstruct the hidden
instances or launches; such use is a shortcut, not a method. Readouts must be
finite with entries of magnitude at most 1e6; a single call may take at most
600 s.

Not measured: training cost, energy, electrical power or silicon fidelity.
