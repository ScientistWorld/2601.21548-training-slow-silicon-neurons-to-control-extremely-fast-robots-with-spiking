# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper adapts neuromorphic reinforcement learning from Pong-like games to a fast robotic air-hockey interception task. Its central contribution is an online spiking controller: a fixed random reservoir with local readout learning, which learns precise motion-primitive timing from a small number of episodes.

## Approach

- **Encoding:** each of the six state variables is encoded by a Gaussian population code of 10 cells, giving 60 deterministic input spike generators.
- **Hidden layer:** the generators drive 1020 adaptive LIF silicon neurons on DYNAP-SE through fixed random feedforward synapses.
- **Readout:** filtered hidden spikes feed two linear readout units. A softmax over them gives the probabilities of the home and forward primitives.
- **Learning:** only the hidden-to-readout weights learn. They are updated every two trials with the e-prop-style rule (Eq. 1): the global reward times an eligibility trace, i.e. the discounted sum of (action probability - chosen-action indicator) x filtered presynaptic activity.

## Reproduction in This Gym

- **Code:** `method/spiking_eprop_rl.py` and `method/train.py`.
- **Reservoir:** random sparse excitatory/inhibitory input synapses with per-seed device mismatch, and a global synaptic gain calibrated so that about 25% of neuron-state pairs are driven above threshold.
- **Readout learning:** Eq. 1 exactly, every two trials, for 64000 episodes per run and 10 reservoir seeds per experiment. The learning rate is expressed relative to the readout input power of the untrained reservoir. Its value is 0.012. The earlier visible-slice selection used the simpler kinematic simulator and is not evidence of convergence on the final robot. The extended trial budget addresses unfinished learning on the software robot, including one poorly converged chip at 32000 trials; it is 32 times the paper's 2000 trials, with the learning rule unchanged.
- **Encoding ranges:** the velocity code spans the experiment's speed range, so wider ranges get coarser resolution.
- **Evaluation:** actions are sampled from the softmax, as in the paper. Fixed held-out rollouts measure final performance rather than the paper's online learning-curve average.
- **Single-core experiment:** the same recipe with 256 neurons.

## Main Claims

Primary claim: a compact neuromorphic spiking RL controller can learn robust high-speed robotic air-hockey interception, reaching 96-98% success in randomized dynamic settings.

Secondary claims:
- Wider input velocity ranges slow learning and slightly reduce asymptotic performance (97% for [0.7, 0.9] m/s, 93% for [0.7, 1.5] m/s, 86% for [0.5, 2.0] m/s).
- Learned behaviour consolidates from stochastic action switching into reliable timing.
- The approach uses about 10x fewer neurons than a prior bio-inspired air-hockey controller.
