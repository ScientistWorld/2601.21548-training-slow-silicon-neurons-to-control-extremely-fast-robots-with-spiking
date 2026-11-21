# Overview

- **Paper ID:** 2601.21548
- **Title:** Training slow silicon neurons to control extremely fast robots with spiking reinforcement learning
- **Domain:** Neuromorphic reinforcement learning / robotics
- **TL;DR:** The paper trains a compact spiking controller online, within about 2000 trials, to time air-hockey interception primitives under fast puck dynamics.

## Short Summary

The work asks whether neuromorphic reinforcement learning can scale from simple game benchmarks to real-time robotic air hockey. The controller observes a 6D continuous state, encoded as spikes into a fixed random spiking substrate; only a two-action readout learns, online, from reward. The reported system reaches high success under randomized puck speeds and launch positions with 1020 hidden neurons and about 2000 training episodes.

## Key Results

- Randomized speed and position: success stabilizes above 98% after about 1300 episodes.
- Randomized speed alone: above 96% after about 1500 episodes.
- Velocity-range tests degrade gracefully: above 97% for narrow ranges, about 93% for [0.7, 1.5] m/s, and 86% for [0.5, 2.0] m/s.

## This Gym

- The scored quantity is the paper's: on-line success while learning within the
  paper's 2000-trial budget, and the success reached by its end.
- Submissions are learning procedures. The evaluator runs them online on 20
  fixed random controller instances that exist only in the hidden test.
- Three scored settings are the paper's own: randomized launch position and
  speed, the wide speed range, and the widest speed range.
- Software substitutes: a torque-limited three-joint MuJoCo arm and a software
  spiking substrate; the authors' silicon and exact robot are not reproduced.
