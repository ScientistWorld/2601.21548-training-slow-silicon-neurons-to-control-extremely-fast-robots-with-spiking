The tracked MJCF is a planar three-joint robot, following the open model
structure of MushroomRL's planar air-hockey arm (MIT; license alongside).
Source: https://github.com/MushroomRL/mushroom-rl/tree/818ebf051eefa2f06acfd889e380c293e574f06e/mushroom_rl/environments/assets/air_hockey

This is a documented geometry adaptation, not the authors' unpublished robot:
links are 0.75/0.55/0.40 m so both 0.70/1.50 m targets in Table I are reachable;
joint ranges are +/-3.05 rad and torque caps 80/40/15 Nm. Masses are 3/2/2 kg.
The previous gym had no joints, inertias, actuators or tracking error.
The puck and mallet have the paper's radii. Plane-constrained spheres represent
their circular horizontal contact; there is no vertical flight. Input states,
two task-space cubic primitives, 50 Hz control, 1 kHz physics, reward and
success are as in the paper. The scored launch conditions are the paper's own
(Sec. III). No external model files or meshes are needed. All assets ship in
git; eval/test/robot/ holds the hidden test's frozen copy.

Contact uses MuJoCo's direct stiffness/damping format, `-100000 -20`, with
impedance `0.95 0.99 0.001`. This resolves near-elastic impact at the 1 ms step.
The cancelled audit used `0.002 0.1` in time-constant/damping-ratio format,
which implied stiffness about 25 million and produced unstable accelerations.
See https://mujoco.readthedocs.io/en/3.3.7/modeling.html#solver-parameters.
Both simulator copies reject finite rollout tails that reset or stop time.
Numerical safety bounds also reject puck speeds above 50 m/s, joint speeds
above 100 rad/s, or puck positions over 10 cm beyond a table boundary, before
any goal is scored. These bounds are far above the task's 0.5--2 m/s launches.
