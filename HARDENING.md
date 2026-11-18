# Final hardening

The completed audit rewards the tested scientific improvement and penalizes the
weaker controls and major shortcuts. The primary honest variant improves all
three success rates over the reference. Its normalized rank gain is +0.040983,
with paired 95% bootstrap interval [0.022084, 0.062891] and a positive combined
gain on all ten chips. This is evidence of headroom on this fixed software
problem, not a claim that every possible shortcut has been excluded.

Job 516de1db-87c completed successfully, including scientific checks and
validation. The final canonical score files contain only the reproduced
reference and three evaluator-owned baselines. Variant and shortcut results
are retained in [the audit scores](redteam/final/all_scores_ref64.json),
[paired uncertainty](redteam/final/paired_bootstrap_ref64.json), and
[scientific checks](redteam/final/scientific_checks.json).

## What changed and why

The paper's actual procedure is retained: six observations, 60 deterministic
Gaussian population spike generators, a fixed random feedforward adaptive LIF
substrate, two filtered-spike linear readouts, softmax action sampling, and the
reward-weighted local Eq. 1 update every two trials. There is no return
centering, trajectory-length normalization, probability floor, or learned
hidden recurrence in the reference. The Eq. 2 shaping condition uses
instantaneous Euclidean distance, rather than a contact-history flag.

The earlier task-level mallet approximation is replaced by a torque-limited
three-joint MuJoCo arm with inverse kinematics and cubic task-space primitives.
Physics and neurons advance at 1 ms; actions arrive every 20 ms. Joint inertia,
tracking error and physical contact now determine interception. The software
neurons and robot geometry are documented substitutes for the authors' silicon
and exact robot. This restores the computational training procedure and the
robotic timing problem; it does not reproduce hardware measurements.

The reference uses the exact unaveraged endpoint after 64000 trials per chip,
32 times the paper's 2000 trials. Shorter software-robot runs were still
learning and allowed the hand rule to win. Extending training keeps Eq. 1,
its two-trial update interval, learning rate and discount unchanged. No setup
was changed to approach a published number. Original paper values 0.98, 0.93
and 0.86 remain unchanged under _paper_reported in
[reference.json](scoring/reference.json); they are not local score targets.

REVIEW.md identified saturated easy conditions, seed selection comparable to
honest gains, and nearly perfect hand-written timing. The repairs address all
three:

- Retire the saturated main and wide conditions from rank. Use weights 0.2 for
  extreme speeds, 0.3 for restricted capacity, and 0.5 for sensing robustness.
  Robustness carries half the weight because uncertain temporal control is
  the unresolved scientific problem and clean launches admit simple schedules.
- Add a 1020-unit robustness experiment: launch x in [0.80, 1.35] m, incoming
  heading in [-0.30, 0] rad, position noise 0.035 m, velocity noise 0.35 m/s,
  and two control ticks (40 ms) of observation latency. This extends the
  paper's discussion of uncertainty; it is not a reported paper experiment.
- Freeze the ten supplied chip instances and check every non-readout array
  against evaluator-owned copies. A slot cannot select a lucky reservoir.
- Enlarge evaluation from 1000 to 4096 launches per chip and pair physical
  inputs, observation noise and action draws across methods. Train and test
  use independent inputs of the same difficulty.

## Reference and variants: before and after

REDTEAM.md described an older two-task toy gym: reference 0.960 / 0.674,
honest learned timing 0.996 / 1.000 (rank +0.255275), and no learning
0.59733 / 0.49200 (rank -0.732019). Enumeration also reached 1 / 1 and constant
forward tied the reference, so this ordering did not establish scientific
contribution. Those old controllers and ranks are not numerically comparable
to the new robot and fixed-chip interface.

At the start of this iteration, REVIEW.md's four-task reference was
0.9997 / 0.9992 / 0.9741 / 0.9312 (main / wide / extreme / restricted capacity).
Its approximate seed standard error was 0.017 and selecting seeds paid about
0.03–0.04. The two easiest tasks offered effectively no useful headroom.
The actual pre-change files are preserved in
[scores_test_before.json](redteam/final/scores_test_before.json).

Final held-out results below use the same 4096 inputs on each of ten chips.
Ranks use the complete 17-method audit cohort, the final weights, and population
standard deviations across that cohort. Reference rank is zero by definition;
rank magnitudes change with the future cohort. Calibration's success dominance
in all three experiments preserves its ordering under positive scales.

| Method | Extreme | Restricted capacity | Sensing robustness | Rank change |
|---|---:|---:|---:|---:|
| Reference, Eq. 1, 64000 trials | 0.996973 | 0.990918 | 0.840601 | 0 |
| Public calibration, primary improvement | 0.999756 | 0.998730 | 0.864893 | +0.040983 |
| Fixed readout sharpening | 0.999780 | 0.996191 | 0.848462 | +0.015929 |
| Late readout averaging | 0.996631 | 0.992334 | 0.839429 | -0.000875 |
| Same trajectory at 32000 trials | 0.988452 | 0.976172 | 0.738330 | -0.159054 |
| Weak training, 2000 trials | 0.603931 | 0.593457 | 0.270630 | -1.248404 |
| Untrained | 0.082837 | 0.064478 | 0.186841 | -1.949087 |
| Random action baseline | 0.076245 | 0.075903 | 0.190308 | -1.939451 |
| Constant forward | 0.207764 | 0.201660 | 0.194580 | -1.791772 |
| Best global timing rule | 1.000000 | 1.000000 | 0.725830 | -0.155892 |
| Public rule imitation | 0.999536 | 0.951587 | 0.695044 | -0.232118 |

Canonical reference train success is 0.996899 / 0.991431 / 0.843506, versus
test 0.996973 / 0.990918 / 0.840601. The largest split difference is 0.00291;
the independent slices measure comparable difficulty. Learned reference
control clearly beats random, constant action and untrained control, while
respecting the 1020 / 256 / 1020 unit limits. Restricted capacity reduces clean
success in the paper's direction. This supports the software core claim,
without establishing the paper's 2000-trial sample efficiency or hardware claims.

## Headroom and paired noise

The reference is at the upper bound of none of the scored metrics: the only
ranked metric is success, whose three values are 0.996973, 0.990918 and
0.840601. Clean conditions remain close to saturation. The added, weighted
sensing experiment provides substantial error and distinguishes learning from
short timing rules. Even the best honest variant still fails 13.51% there;
no simple variant reaches all bounds.

The primary alternative, [public calibration](method/calibrate_readout.py),
chooses one global readout blend, inverse temperature and action offset per
experiment. Its fixed 36 candidates use only 256 fresh public launches on each
chip. The selected (blend, temperature, offset) values are (0, 2, -2),
(0.5, 8, -2), and (0.5, 4, -2). Hidden inputs and outcomes are not used for
selection. The reference remains the unmodified Eq. 1 endpoint. Any readout
learning procedure can use the same numeric submission interface.

A predeclared check requires this primary alternative's paired lower bound
above zero. The test-private audit jointly resamples ten chip identities and
whole episode columns in 4096 bootstrap replicates. It shares resampling draws
across methods, preserves within-episode dependence, shares chip identities
across experiments, and holds measured cohort normalization fixed.

| Variant | Paired rank gain | 95% bootstrap interval | Chips with positive combined gain |
|---|---:|---|---:|
| Public calibration | +0.040983 | [0.022084, 0.062891] | 10 / 10 |
| Fixed sharpening | +0.015929 | [-0.001403, 0.034055] | 6 / 10 |
| Late averaging | -0.000875 | [-0.032035, 0.031001] | 4 / 10 |

Calibration also passes the exact one-sided chip sign check, p=0.0009765625.
Sharpening and averaging do not independently establish improvement beyond
noise; they are not substituted for the primary check. Inference remains
limited to the fixed ten chips and the chosen task distribution, rather than
all possible robot or sensor distributions.

## Major shortcuts

Before values here come from REDTEAM.md unless marked REVIEW. Its old two-task
scores and ranks are retained as provenance, not directly comparable effect
sizes. After values are from the final hidden audit. An invalid submission's
complete worst row is success=0, worst-chip success=0, hit=0 and units equal to
the experiment budget, in every experiment; its final audit rank is -2.294264.

| Shortcut and before result | Fix | After result |
|---|---|---|
| Forged summaries: 1 / 1, rank +0.263152; impossible rates: 2 / 2, rank +2.578002 | Ignore all submitted numbers and flags; execute bounded numeric configurations and derive every metric from raw physical outcomes | JSON-only forged submission: 0 / 0 / 0, rank -2.294264. Legacy summary values have no input to scoring |
| Failed task compensated with huge auxiliary metrics: rank +0.387261 | Rank success alone; contact and worst-chip success are diagnostics, capacity is a hard constraint | Forged-metric submission fails every experiment, rank -2.294264; auxiliary coefficients are zero |
| Empty or omitted experiment: free rank tie | Discover every on-disk method directory and require all experiments and ten slots; any failure applies to the whole method | Missing-experiment and JSON-only empty-controller cases: 0 / 0 / 0, rank -2.294264 |
| Training allowlist silently skips another method: free tie | Discover directories independently of train scores, including hidden names; always include the configured reference | All 17 audit methods get all three complete rows; the five invalid methods are retained with worst values |
| Omitted or fictitious capacity: reference tie | Count actual units, validate fields and bounds, enforce assigned chip identity | Over-capacity submission: 0 / 0 / 0, rank -2.294264 |
| Nonfinite values accepted; null input can leave stale scores | Validate numeric arrays and physics; initialize every result with worst values and write complete rows atomically | Nonfinite-weight submission: 0 / 0 / 0, rank -2.294264; no omitted metrics |
| Lucky reservoir selection: +0.03–0.04 (REVIEW) | Freeze ten chips per experiment and compare every non-readout array exactly | Copying seed 0 into all slots with readout jitter: 0 / 0 / 0, rank -2.294264 |
| Constant forward: reference task tie, rank +0.000994 | Robot dynamics and richer launches make timing meaningful | 0.207764 / 0.201660 / 0.194580, rank -1.791772 |
| Hand-written interception nearly solves task; imitation outranks undertrained robot reference | Full paper-sized robustness substrate, richer noisy/latent observations, converged Eq. 1 training and weight 0.5 on robustness | Best of 27 global filtered timing rules: 1 / 1 / 0.725830, rank -0.155892; learned rule imitation: 0.999536 / 0.951587 / 0.695044, rank -0.232118 |
| Enumerating 31 timings and known-seed lookup: both 1 / 1, rank +0.262159 | Hidden launches are unseen; only static readout arrays execute, with no submitted planner, callback, simulator or lookup-file access | Legacy summary-only output is the rejected JSON interface above; its old executables are not meaningful controllers on the new interface. The new global rule bank and public rule imitation both lose |
| Replacing public simulator: random reaches 1 / 1, rank +0.262159; moving baseline was a tie | Test owns independent robot, simulator, chip runtime, inputs and baseline procedures; public mutations cannot change them | Genuine random baseline is 0.076245 / 0.075903 / 0.190308, rank -1.939451. No new public-mutation replay is claimed; the public module is absent from test imports |

These are interface and task repairs, rather than defenses against leftover
processes or interpreter changes. The new audit replays forged metrics,
missing experiments, capacity violation, nonfinite weights, copied chips,
constant action, timing rules and imitation. Legacy executables that merely
write the removed JSON schema are represented by the replayed JSON-only
attack; their former perfect summaries are no longer runnable policies.

## Scoring and isolation

Only per-experiment results in scoring/scores_test.json and
scoring/scores_train.json are ranked. Audit ranks, verdicts and files elsewhere
have no ranking authority. [eval/rank.py](eval/rank.py) sums experiment weight
times the absolute coefficient times better-direction improvement over the
reference, divided by population standard deviation across ranked attempts.
An identical column contributes zero. Success is the sole ranked quantity;
a completely failed task cannot buy rank through diagnostics or declared size.

All test input generation, seeds and raw per-episode answers are confined to
eval/test/. No method code runs in the hidden test: bounded numeric readouts
are loaded into its own runtime. Thus a method cannot read test answers or
privately evaluate hidden candidates during inference. Readout training remains
open to alternative optimizers, credit assignment, exploration, value estimates,
longer training and public-data imitation. The restriction does not limit a
submission to a menu of the reference's settings.

The two split evaluators share no imported implementation, and eval/ and data/
never import method/. Everything on the other side of each comparison is
owned by eval/: physical task, baseline operating points, chip copies, inputs,
thresholds, capacities and test size. Test scoring never retrains. It runs
fixed-sized rollouts without a wall-clock stopping rule or wall-clock metric.
All training and calibration workers had exited before serial reference and
baseline scoring. Saved, source-validated raw outcomes were rescored where
only packaging/evaluation changed; no method-written metric was trusted.

## Rejected checks and operational repairs

Early unstable MuJoCo runs were discarded. Explicit contact stiffness and
damping now fit the 1 ms timestep. Nonfinite states, resets, stalled time,
puck speed over 50 m/s, joint speed over 100 rad/s, or penetration over 10 cm
fail before a goal can count. The final job passed public physics preflight,
exact interrupted/uninterrupted learning checks and concurrent atomic-write
checks. Every controller affected by physical or training changes was rerun.

The 2000-trial robot reference scored 0.53442 / 0.48931 / 0.21348; 8000 trials
scored 0.92188 / 0.92441 / 0.65393. Rule imitation still won that rejected setup
(0.99993 / 0.98975 / 0.75591). On the final physical task the 32000-trial
reference scored 0.988452 / 0.976172 / 0.738330, but the timing bank still
narrowly won (rank +0.002584). Its longer-training and averaging improvements
had conservative normal lower bounds -0.057016 and -0.047787: one robustness
chip had learned very late. Neither audit was certified. Their aggregate
results remain in rejected_robot_setup/ and reference32_audit/ under redteam/final/.
The final reference promotion uses already-completed, exact trial-64000
checkpoints, with no change to the training trajectory. Calibration was then
selected on public data before its hidden run.

Worker drains caused repeated job restarts and one lost-node exit. They did
not justify changing scientific outputs. Exact training/RNG checkpoints and
audit-only raw rollout checkpoints preserve progress. Raw cache keys cover
full inputs, controller arrays, evaluator sources and pinned library versions;
normal no-argument evaluators are uncached. A persistence-only repair replaced
colliding temporary filenames with unique atomic writes in both independent
evaluators. Exact source migration digests allow only those equivalent old
raw rollouts to be reused. No time-based training truncation was added.

## Package and remaining limitations

The final script scripts/run.sh again reproduces the reference end to end,
then validates. scripts/evaluate_test.sh takes no arguments and scores every
method on disk. Both canonical slices and scores.json were regenerated from
the final reference and fixed baselines alone; variants are archived outside
outputs/. The container builds from environment/, with pinned NumPy and MuJoCo
installed by setup.sh. All small authored chip assets are tracked. No external
pre-existing dataset or model is used, so download.sh has no remote artifact to
fetch. Committed code has no shared/ references. Briefing and scoring documents
describe the final problem and interface. python validate.py passes; its four
heuristic RNG warnings do not indicate detached scoring: the evaluators load
and execute numeric controller files, and raw outcomes change with the method.

Remaining issues are explicit:

- Selecting among repeated training runs on an assigned chip, undisclosed
  public-data computation, and copying a stored learned readout are provenance
  questions for the separate attempt grader. The evaluator cannot verify an
  asserted learning history. Fixed chip identities close reservoir selection,
  not all selection among learned readouts.
- Short clean-launch schedules still solve the two clean conditions. They lose
  overall on the weighted robustness experiment, as does public rule imitation.
  Stronger hand-coded or distilled policies were not exhaustively excluded.
- Private search over public training launches remains possible and can improve
  generalization; there is no test-instance search interface. The gym measures
  final control success at fixed capacity, not training efficiency or cost.
- The ten-chip software model and substituted robot do not establish exact
  silicon behavior, electrical power, hardware real time, the paper's 2000-trial
  efficiency, or all of its empirical settings. Hardware fidelity remains
  unmeasured. The milestone is core_claim on the documented software task.
