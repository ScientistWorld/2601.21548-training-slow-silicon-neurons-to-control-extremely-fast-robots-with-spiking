# Problem Definition

## Research Question

Can a low-capacity controller learn to intercept a fast puck in a continuous robotic air-hockey task from online interaction? The agent observes a 6D continuous state: puck position and velocity plus end-effector position. It must choose between two discrete motion primitives at 50 Hz so the end effector reaches the puck at the correct time and sends it forward across the target line.

## Why It Matters

Fast robotic manipulation stresses sample efficiency, timing precision, and compact controllers. A good solution would show that always-on learning controllers can handle real-time physical interaction instead of only slow or forgiving toy domains.

## Success Criteria

A successful system should reach a high episode success rate under randomized puck initial positions and speeds, make contact with the puck reliably, and do so under the fixed small-controller capacity used by the benchmark.
