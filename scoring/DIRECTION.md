# Research Direction

## Research Problem
This benchmark studies online learning for fast robotic interception with compact controllers. The goal is to improve timing and robustness in the same two-primitive air-hockey task, not to solve a different robotics problem.

## Core Contribution
Improvements should address the same closed-loop learning problem: a compact controller receives the benchmark state, chooses among the benchmark primitives, and learns from task interaction under the same evaluation protocol.

## Approach Scope
Scientists may alter state representations, controller dynamics, exploration, credit assignment, and update rules. These changes remain in scope when the system still uses the same observation/action interface and respects the compact-controller scoring constraint.

## Out of Bounds
Replacing the controller with model-predictive control, a hand-coded intercept-time planner, a large deep RL policy, imitation learning from an oracle, or privileged simulator-state planning is out of bounds. Those approaches may solve the task but do not test the benchmark's compact online-learning claim.
