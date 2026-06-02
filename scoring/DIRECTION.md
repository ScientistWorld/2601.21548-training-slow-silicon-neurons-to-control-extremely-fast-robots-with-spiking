# Research Direction

## Research Problem
This benchmark studies online learning for fast robotic interception with compact event-driven controllers. The goal is to improve timing and robustness in the same two-primitive air-hockey task, not to solve a different robotics problem.

## Core Contribution
The paper's contribution is spike-based reinforcement learning with local readout adaptation over a fixed random reservoir. Improvements should stay within compact spiking or event-driven reservoir-control methods for the same state/action interface.

## Approach Scope
Scientists may alter encoding ranges, reservoir connectivity, neuron dynamics, trace filtering, exploration, reward-to-update credit assignment, and local readout learning. These changes remain in scope when the system is still an online event-driven controller for the same task.

## Out of Bounds
Replacing the controller with model-predictive control, a hand-coded intercept-time planner, a large deep RL policy, imitation learning from an oracle, or privileged simulator-state planning is out of bounds. Those approaches may solve the task but do not test the paper's neuromorphic learning claim.
