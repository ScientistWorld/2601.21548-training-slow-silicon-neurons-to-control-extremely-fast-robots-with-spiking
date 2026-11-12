# Problem Definition

## Research Question

Can a compact controller running on slow, analog-like spiking hardware learn, from online interaction, to time a fast robotic interception? A robot plays air hockey on a 1.948 m x 1.038 m table. A puck is launched sideways across the table near x = 1.0 m, at up to 2 m/s, and bounces off the side walls. Fifty times per second the controller observes a 6D state: puck position and velocity, plus end-effector position. It then picks one of two motion primitives. Action 0 moves the end effector toward a home point at x = 0.70 m; action 1 moves it toward a forward point at x = 1.50 m. Each primitive is executed open loop as a smooth spline from the end effector's current position and velocity. The controller must choose *when* to strike so that the mallet meets the puck mid-stroke and drives it past x = 1.5 m with forward velocity. Striking too early or too late misses.

## Why It Matters

Robotic interception needs decisions committed tens of milliseconds ahead of contact, and timing that depends on the puck's position and speed. Neuromorphic processors promise always-on learning at milliwatt power. But they offer few, slow, imprecise neurons and a limited input resolution. Showing that such a substrate can learn precise timing from a modest number of trials, and characterising where capacity and input range limit it, is what moves neuromorphic control beyond simple game benchmarks.

## Task Settings

| Setting | Puck launch position (x) | Launch speed | Controller budget |
|---|---|---|---|
| Randomized position and speed | 0.95-1.05 m | 1.0-1.5 m/s | 1020 hidden units |
| Wide speed range | 1.00 m | 0.7-1.5 m/s | 1020 hidden units |
| Extreme speed range | 1.00 m | 0.5-2.0 m/s | 1020 hidden units |
| Extreme speed range, single core | 1.00 m | 0.5-2.0 m/s | 256 hidden units |

The physics runs at 1 ms with frictionless puck motion and elastic collisions. An episode lasts at most 100 control steps (2 s), so a missed puck may come back after a wall bounce. The reward signal available during learning is +20 when the puck crosses x > 1.5 m with v_x > 0.1 m/s, which ends the episode. A step with mallet-puck contact earns 0.2 x v_x. Every other step costs -0.1.

## Success Criteria

A successful system reaches high episode success on unseen launches in every setting. It does so reliably across independent training runs and within the hidden-unit budget of each setting. Wider speed ranges and smaller budgets are expected to cost performance; reducing that cost is the open problem.
