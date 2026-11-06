# Training slow silicon neurons to control extremely fast robots with spiking reinforcement learning

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2601.21548"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2601.21548-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2601.21548](https://arxiv.org/abs/2601.21548) |
|:---|:---|
| **Domain** | Neuromorphic reinforcement learning / robotics |
| **TL;DR** | The paper trains a compact spiking neural controller with local readout learning to time air-hockey interception primitives under fast puck dynamics. |

## Summary

The work studies whether neuromorphic reinforcement learning can scale from simple game benchmarks to real-time robotic air hockey. The controller observes a 6D continuous state, encodes it as spikes, processes it through a fixed random reservoir, and learns a two-action readout online. The reported system reaches high success rates under randomized puck speeds and initial positions while using only 1020 hidden neurons.

## Problem

Can a low-capacity controller learn to intercept a fast puck in a continuous robotic air-hockey task from online interaction? The agent observes a 6D continuous state: puck position and velocity plus end-effector position. It must choose between two discrete motion primitives at 50 Hz so the end effector reaches the puck at the correct time and sends it forward across the target line.

## The paper's method

The paper adapts neuromorphic reinforcement learning from Pong-like games to a fast robotic air-hockey interception task. Its central contribution is an online spiking controller with a fixed random reservoir and local readout learning that can learn precise motion-primitive timing from a small number of episodes.

## Key results of the paper

- Randomized speed and position setting stabilizes above 98% success after about 1300 episodes.
- Randomized speed alone reaches above 96% success after about 1500 episodes.
- Velocity-range tests show graceful degradation from above 97% success on narrow ranges to about 93% on the widest main range.

## Run it

```bash
pip install sciworld
sw research 2601.21548 --mode innovate --agent claude-code
sw evaluate 2601.21548
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

