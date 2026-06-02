# Overview

- **Paper ID:** 2601.21548
- **Title:** Training slow silicon neurons to control extremely fast robots with spiking reinforcement learning
- **Domain:** Neuromorphic reinforcement learning / robotics
- **TL;DR:** The paper trains a compact spiking neural controller with local readout learning to time air-hockey interception primitives under fast puck dynamics.

## Short Summary

The work studies whether neuromorphic reinforcement learning can scale from simple game benchmarks to real-time robotic air hockey. The controller observes a 6D continuous state, encodes it as spikes, processes it through a fixed random reservoir, and learns a two-action readout online. The reported system reaches high success rates under randomized puck speeds and initial positions while using only 1020 hidden neurons.

## Key Results

- Randomized speed and position setting stabilizes above 98% success after about 1300 episodes.
- Randomized speed alone reaches above 96% success after about 1500 episodes.
- Velocity-range tests show graceful degradation from above 97% success on narrow ranges to about 93% on the widest main range.
