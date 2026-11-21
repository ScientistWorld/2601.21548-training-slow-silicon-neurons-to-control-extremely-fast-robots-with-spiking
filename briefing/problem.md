# Problem

Learn, online and within a small number of trials, to intercept a fast
air-hockey puck with a fixed compact spiking controller.

**Task.** A puck launched from (x0, 0.45 m) slides laterally toward the centre
line of a 1.948 x 1.038 m table and rebounds from the side walls. Every 20 ms the
controller selects one of two motion primitives; the robot then executes a cubic
task-space trajectory toward (0.70, 0) m (home) or (1.50, 0) m (forward). Joint
inertia, torque limits and tracking error make the timing of these commitments
matter. Physics runs at 1 kHz. An episode succeeds when the puck passes
x > 1.5 m with forward speed above 0.1 m/s, and lasts at most two seconds.
The reward per 20 ms step is +20 for success, 0.2 times the puck's forward speed
when puck and mallet are within 0.08 m, and -0.1 otherwise.

**Controller.** The six observed variables (puck position and velocity, mallet
position) are encoded by a fixed population code into 60 spike generators that
drive 1020 fixed, randomly connected spiking neurons (a "chip"). The chip is
not trainable and its synapses are not visible. What can be learned is a linear
readout of the chip's filtered spike trains: two action scores whose softmax
gives the probability of each primitive at every step.

**Learning budget.** On each chip the learner gets 2000 sequential episodes,
starting from no knowledge of that chip. After each episode it sees what that
episode produced on its chip (the filtered spike trains at each step, the action
probabilities used, the sampled actions and the rewards) and may change the
readout before the next episode. It never sees robot states, launches or the
chip's parameters.

**Settings (the paper's own).** Randomized launch position (0.95–1.05 m) and
speed (1.0–1.5 m/s); launch speeds 0.7–1.5 m/s; launch speeds 0.5–2.0 m/s. In
the last two the population code spans a wider velocity range with the same
cells, so it resolves velocity more coarsely.

**Research question.** How quickly, and how reliably across chips, can a fixed
random substrate's readout be learned online so that interception succeeds?
The software robot and spiking substrate are substitutes for the authors'
robot and silicon; this gym does not measure energy or hardware real time.
