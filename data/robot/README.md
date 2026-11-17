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
success are as in the paper. The explicit sensor-noise condition is a new
robustness extension motivated by Section IV, not a paper-reported result.
No external model files or meshes are needed. All assets ship in git.
