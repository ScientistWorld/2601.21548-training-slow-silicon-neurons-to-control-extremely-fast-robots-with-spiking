# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper adapts neuromorphic reinforcement learning from Pong-like games to a fast robotic air-hockey interception task. Its central contribution is an online spiking controller with a fixed random reservoir and local readout learning that can learn precise motion-primitive timing from a small number of episodes.

## Approach

The controller encodes six continuous state variables with Gaussian population codes, producing 60 input spike streams. These drive 1020 fixed random silicon AdEx/LIF neurons on DYNAP-SE. Filtered hidden spikes feed two linear readout units whose softmax probabilities select between home and forward motion primitives. Hidden-to-readout weights are updated every two trials using an e-prop-style policy-gradient rule that combines filtered presynaptic activity, action-probability error, discounted reward, and a global reward signal.

## Main Claims

Primary claim: a compact neuromorphic spiking RL controller can learn robust high-speed robotic air-hockey interception, reaching 96-98% success in randomized dynamic settings.

Secondary claims: wider input velocity ranges slow learning and slightly reduce asymptotic performance; learned behavior consolidates from stochastic action switching into reliable timing; the approach uses about 10x fewer neurons than a prior bio-inspired air-hockey controller.
