# Research Direction

## Research Problem
The benchmark studies online learning of fast interception timing on a compact spiking substrate. The aim is to raise success and run-to-run reliability in the hardest settings: broad puck-speed ranges, and a quarter-size controller. The same interface and budgets apply throughout.

## Core Contribution
Improvements should come from a better learner or a better use of the substrate. Examples:
- input encodings that resolve the timing-relevant state more finely
- connectivity or neuron parameters that make the hidden representation easier to read out
- credit assignment, exploration or stability measures that make every independent training run converge

## Approach Scope
Encodings, hidden connectivity (including recurrence), neuron parameters, readout learning rules, training length and stabilisation are all in scope. The evaluator scores the resulting configuration on held-out launches.

## Out of Bounds
These may score well but do not test the benchmark's claim of compact, online learning:
- Distilling per-episode plans obtained by simulating candidate strike times into the network
- Hand-deriving the intercept rule from the simulator's equations and writing it into the weights
- Imitating any oracle
- Training more independent runs than are submitted and keeping the best ones
