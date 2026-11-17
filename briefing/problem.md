# Problem

Learn to intercept a fast air-hockey puck with a compact controller. Every
20 ms the controller receives puck position and velocity and mallet position
(six numbers), and selects a homeward or forward motion primitive. The robot
executes a cubic trajectory toward (0.70, 0) or (1.50, 0) metres. Joint inertia,
torque limits and tracking error make the timing of this commitment matter.
Physics runs at 1 kHz on a 1.948 by 1.038 metre table.

An episode succeeds when the puck passes x > 1.5 m with forward speed > 0.1 m/s.
It lasts at most two seconds. Learning receives +20 for success, 0.2 times
forward puck speed within 0.08 m of the mallet, and -0.1 otherwise. The mallet
and puck radii are 0.04815 and 0.03165 m.

The scored settings cover speeds 0.5–2.0 m/s with 1020 or 256 controller units,
and a 1020-unit robustness setting with launch positions 0.80–1.35 m,
headings tilted up to 0.30 rad toward the near end, noisy observations and
40 ms observation latency. The first two settings launch laterally from x=1 m.
The last setting extends the paper's discussion of sensing uncertainty.

The research question is whether learning improves success across unseen
launches while respecting a fixed compact substrate. Training procedures are
open; every method uses the same supplied ten physical controller instances.
The software robot and controller are substitutes for the authors' robot and
silicon. This gym does not measure electrical energy or hardware real time.
