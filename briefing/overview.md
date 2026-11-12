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

- **Substitute task:** a task-level air-hockey simulator in the paper's geometry: lateral puck launch, spline motion primitives, 1 ms physics.
- **Substitute hardware:** a software spiking substrate standing in for the DYNAP-SE chip.
- **What is scored:** submitted controller configurations, executed by the evaluator on held-out episodes. There are four experiments, including a 256-unit "single core" setting where capacity limits performance.
