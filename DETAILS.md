# Final gym guide

The paper is [Training slow silicon neurons to control extremely fast robots
with spiking reinforcement learning](paper/2601.21548.md). The gym implements
its Eq. 1 readout update on a documented software spiking substrate and a
MuJoCo three-joint robot. It evaluates successful interception at fixed
controller capacity on unseen launches.

The final three experiments cover broad puck speeds, restricted capacity, and
noisy delayed sensing with richer launches. All methods use the same ten
assigned chip instances. They supply numeric readout weights; the independent
evaluator runs the controllers and derives success from physical trajectories.
The reproduced reference uses 64000 trials per chip, versus the paper's 2000.
This is a software core-claim reproduction, without hardware or sample-efficiency
claims.

- [Problem](briefing/problem.md) and [evaluation contract](briefing/evaluation.md)
  describe the scientific task and submission format.
- [Paper method and local departures](briefing/method.md) document the actual
  training procedure.
- [Experiment purposes](scoring/EXPERIMENTS.md), [targets](scoring/TARGETS.md),
  [constraints](scoring/CONSTRAINTS.md), and [research scope](scoring/DIRECTION.md)
  describe the scored surface.
- [Reference](scoring/reference.json) preserves the original paper numbers
  separately from measured software reference and baseline results.
- [Hardening report](HARDENING.md) records before/after scores, paired headroom,
  shortcut repairs, rejected checks and remaining limitations.

Run the installed container's reference pipeline with bash scripts/run.sh.
Run bash scripts/evaluate_train.sh and bash scripts/evaluate_test.sh to score
every method directory under outputs/. Only each experiment's results in the
canonical scoring/scores_train.json and scoring/scores_test.json are ranked.
