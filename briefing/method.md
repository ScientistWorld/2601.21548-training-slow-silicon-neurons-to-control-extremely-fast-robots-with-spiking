# Paper's Method

## Key Contribution

The paper adapts neuromorphic reinforcement learning from Pong-like games to a
fast robotic air-hockey interception task: an online spiking controller with a
fixed random substrate and local readout learning, which learns precise
motion-primitive timing within about 2000 episodes.

## Approach

- **Encoding:** each of the six state variables is encoded by a Gaussian
  population code of 10 cells (60 deterministic spike generators).
- **Hidden layer:** 1020 adaptive LIF silicon neurons on DYNAP-SE with fixed
  random feedforward synapses from the generators.
- **Readout:** filtered hidden spikes feed two linear readout units; a softmax
  gives the probabilities of the home and forward primitives.
- **Learning:** only hidden-to-readout weights learn, every two trials, with
  the e-prop rule (Eq. 1): dw_ik = -alpha sum_t r^t sum_{t'<=t} gamma^(t-t')
  (pi_k^t' - 1[a^t'=k]) sbar_i^t'.

## Reproduction in This Gym

- **Code:** `method/eprop_learner.py` implements Eq. 1 exactly as an online
  learner (zero initial readout, update every two trials, discount 0.95, no
  bias learning, no baseline or normalization of returns). alpha is
  `learning_rate / P`, with P the mean squared trace norm of the first two
  (random) trials, so one value transfers across chips.
- **Learning rate:** the paper does not report alpha. `method/select_learning_rate.py`
  sweeps it on the public chips with the paper's 2000-trial budget and keeps
  the best (0.012); see `method/selected_hyperparameters.json`.
- **Substrate:** `data/neuromorphic_chip.py` (1 ms LIF dynamics with synaptic
  filtering, adaptation, refractoriness and per-chip mismatch; global gain
  calibrated so about 25% of neuron-state pairs are driven above threshold;
  30 ms readout trace filter). The velocity code spans each setting's speed range.
- **Install:** `scripts/method.sh` writes `outputs/spiking_eprop/learner.py` and
  `config.json`; the evaluators run it online.

## Main Claims

Primary: a compact neuromorphic spiking RL controller learns robust high-speed
robotic interception online within about 2000 trials, reaching 96–98% success
in randomized dynamic settings.

Secondary: wider input velocity ranges slow learning and reduce asymptotic
success (97% for [0.7, 0.9] m/s, 93% for [0.7, 1.5] m/s, 86% for [0.5, 2.0] m/s);
behaviour consolidates from stochastic switching into reliable timing; about
10x fewer neurons than a prior bio-inspired controller.
