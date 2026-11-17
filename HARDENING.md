# Final hardening

This iteration addresses REVIEW.md as well as REDTEAM.md. The prior report,
reference definition and stored scores are preserved in `redteam/final/`.
The final combined compute audit is in progress; its stable training phase is
complete, but scoring is incomplete. No final post-fix empirical claims are
made in this draft. Results will be filled after the resumed audit finishes.

The old kinematic simulator let an analytic timing rule bypass robot control.
It is replaced by a torque-limited three-joint MuJoCo robot: inverse kinematics
and cubic task-space primitives now drive real joint and contact dynamics.
The geometry is an explicit substitute, not the authors' exact robot. The
software neuron model remains a substitute for silicon. Eq. 1, Gaussian spike
encoding, fixed random feedforward reservoir, two-trial updates and 2000-trial
reference training are retained.

Chip instances are fixed per slot and checked against test-owned copies.
Changing a random seed or copying one lucky chip into all slots now fails,
including tiny readout jitter. Learning procedures remain open on every assigned
chip. Longer training, a value-baseline variant, a weak checkpoint and rule
imitation are included in the combined audit on identical launches.

Only task success is ranked. Contact, worst-chip success and counted units
remain diagnostic. This removes auxiliary compensation for a failed task.
Any invalid experiment fails all experiments with complete worst-value rows.
No missing value is skipped. Dot/underscore method directories are discovered.
Metric direction, absolute coefficients and cohort standardization are explicit
in `eval/rank.py`.

Old saturated headline and wide-range experiments are retired from ranking.
Weights are 0.2 for extreme speeds, 0.3 for restricted capacity and 0.5 for the
new sensor robustness setting. This concentrates rank on limitations rather
than ceiling reproduction. Ten chips now see 4096 paired episodes rather than
1000. The added setting varies launch position and direction, position/velocity
noise, and observation latency. These are robustness extensions, not paper
reported experiments. The original paper numbers are preserved verbatim under
`_paper_reported` in `scoring/reference.json`.

The test has separate frozen simulator, runtime, robot and backbone assets.
Only numeric readouts cross its interface; private per-test search cannot run.
Answers and paired raw outcomes stay inside `eval/test/`. All scored numbers
come from physical rollouts, never method-written metrics or validity flags.
Reference and baselines will be rerun from final code, then rescored alone
when all training workers have exited. No wall-clock threshold is used.

Left to the separate attempt grader: provenance and training-run selection on
an assigned chip. We enforce the chip identities, not an unverifiable training
history. Search or imitation during training is expressible and audited; it
is not excluded merely because it differs from the reference. The gym does
not measure electrical power, real chip performance or exact robot fidelity.

The audit also exports an averaged late-training readout as an honest variance
reduction variant. It follows the same training trajectory as the reference
through episode 2000; only the alternative's final readout is averaged over
later updates. The reference checkpoint is unaffected.

The direct analytic baseline reports the strongest global rule from a fixed
nine-rule bank (eight lead times plus a distance-based travel-time estimate).
It has an intentional selection advantage over a single preselected rule;
there is no per-instance best-candidate oracle. This avoids testing only a
poorly tuned handwritten schedule. Rule imitation tunes on public data only.

The first combined audit was cancelled after repeated MuJoCo QACC instability
warnings. Its partial controllers and calibrated chips are excluded from all
results. The contact time-constant parameterization implied stiffness about
25 million at a 1 ms physics step. Both robot copies now use explicit stiffness
100000 and damping 20. Both simulators also reject non-advancing or reset
rollout times, including finite arrays returned after instability. A public
4096-episode random/switching/rule stability preflight precedes chip calibration
in the replacement combined job. No score from the cancelled run is valid.
Loose velocity and table-boundary safety limits reject numerical explosions
before a spurious goal could mask a subsequent solver warning. Training
failures now stop the audit rather than silently continuing to scoring.

The stable combined job completed all training, then encountered repeated
worker drains. Each restart began scoring from the first method again; it was
cancelled to fix this scheduling failure. The audit now checkpoints individual
raw controller rollouts inside the appropriate split. Keys cover exact
controller arrays, initial conditions, evaluator sources and library versions.
Every resumed score is recomputed from those raw outcomes. A one-off migration
reuses the nine complete experiments from the cancelled stable run only after
verifying its evaluator sources are byte-identical. The normal no-argument
evaluators do not use these audit caches. The scientific setup and trained
controllers are unchanged, so training is not repeated.
