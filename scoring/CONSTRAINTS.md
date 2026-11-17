# Constraints

Each method supplies ten controller files for every experiment. The seed slot
identifies a supplied, calibrated chip; only its two readout arrays (`w_out`,
`b_out`) may change. Wrong instance, altered backbone, nonfinite or out-of-range
values, excessive capacity and missing files fail the entire submission.
Failure receives a complete worst-value row in every experiment.

Any procedure for learning the readout is allowed, including alternative
optimizers, value estimates, more training and imitation. This scope permits
new learning methods rather than just choices among reference hyperparameters.
Evaluation runs the fixed runtime and numeric readout alone: no per-instance
planning code, alternate simulator, lookup file, or submitted metric is used.
The six observed state variables are the only inputs. The two primitives are
the only actions. Test episodes and answers stay inside `eval/test/`.

Selecting different reservoir samples is mechanically prevented. Selecting
among repeated training runs on an assigned chip remains a provenance question
for the separate attempt grader; submit one trained result per assigned slot
and disclose the training budget. The gym measures success and capacity;
it does not claim to measure training cost, power or silicon fidelity.
