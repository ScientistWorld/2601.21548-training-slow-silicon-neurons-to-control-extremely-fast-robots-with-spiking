# Overview

- **Paper ID:** 2601.21548
- **Title:** Training slow silicon neurons to control extremely fast robots with spiking reinforcement learning
- **Domain:** Neuromorphic reinforcement learning / robotics
- **TL;DR:** The paper trains a compact spiking neural controller with local readout learning to time air-hockey interception primitives under fast puck dynamics.

## Short Summary

The work asks whether neuromorphic reinforcement learning can scale from simple game benchmarks to real-time robotic air hockey. The controller observes a 6D continuous state and encodes it as spikes. A fixed random reservoir processes the spikes, and a two-action readout learns online. The reported system reaches high success under randomized puck speeds and launch positions with only 1020 hidden neurons.

## Key Results

- In the randomized speed-and-position setting, success stabilizes above 98% after about 1300 episodes.
- With randomized speed alone, success reaches above 96% after about 1500 episodes.
- Velocity-range tests show graceful degradation: above 97% for narrow ranges, about 93% for the widest main range, and 86% for [0.5, 2.0] m/s.

## This Gym

- Software substitute: a torque-limited three-joint MuJoCo arm and a software
  spiking substrate; the authors' silicon and exact robot are not reproduced.
- Three scored settings: broad puck speeds, restricted controller capacity,
  and a noise/latency extension motivated by the paper's robustness discussion.
- Every method learns readouts for the same ten assigned controller instances.
- The evaluator recomputes closed-loop success on unseen, paired launches.
