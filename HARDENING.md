# Final hardening (audit pending)

The reports identified paying shortcuts and inadequate headroom. Final
verification is still pending. All 30 training trajectories are complete,
including exact 32000- and 64000-trial checkpoints. Job 8a5b7d7e-e69 completed
the entire 32000-trial-reference comparison, then lost its worker node during
canonical train scoring. No application or numerical failure was reported.
The completed comparison **failed scientific certification**: the timing bank
narrowly outranked that reference, and the conservative Gaussian uncertainty
bound did not certify the honest improvement. Neither failure is presented as
a successful hardening.

The reference now uses the already-executed, unaveraged 64000-trial Eq. 1
checkpoints. A public-validation readout-calibration procedure is the
predeclared primary headroom variant; its settings are chosen without hidden
inputs or results. Fixed readout sharpening and late averaging are secondary
comparisons. The continuation reuses source-validated raw outcomes and finishes
isolated reference/baseline scoring. Headroom is not yet certified.

Completed held-out success rates are:

| Controller | Extreme / restricted capacity / sensing robustness |
|---|---|
| Rejected reference, 32000 trials | 0.98845 / 0.97617 / 0.73833 |
| Final reference, 64000 trials | 0.99697 / 0.99092 / 0.84060 |
| Averaged late readout | 0.99663 / 0.99233 / 0.83943 |
| Rule imitation | 0.99954 / 0.95159 / 0.69504 |
| Weak control, 2000 trials | 0.60393 / 0.59346 / 0.27063 |
| Untrained | 0.08284 / 0.06448 / 0.18684 |
| Constant forward | 0.20776 / 0.20166 / 0.19458 |
| Best global timing-bank member | 1.00000 / 1.00000 / 0.72583 |
| Public-validation calibration | Pending hidden scoring |

Forged metrics, a missing experiment, nonfinite weights, excess capacity and
copying one chip's weights into all slots each received zero for every success,
worst-chip success and hit metric in all three experiments, with the full
allowed unit count as their worst constraint value. These are completed
comparisons, not values inferred from unfinished score entries.

Against the 32000-trial reference, normalized ranks were +0.170571 for longer
training, +0.169600 for averaging, +0.002584 for the timing bank, -0.078179 for
rule imitation, -1.155063 for weak training, and -2.253948 for each invalid run.
The summed conservative normal lower bounds were -0.057016 and -0.047787 for
the two honest variants. One robustness chip had only 0.12354 reference success
at 32000 trials; its late improvement produced a strongly skewed distribution
of gains. The full comparison and both uncertainty diagnostics are retained.

Using the completed 64000-trial results as the reference, in that same old
cohort the timing bank's relative rank is -0.167987 and imitation's -0.248749.
These are diagnostics, not final cohort ranks: final ranks must also include
the new variants and use that cohort's standard deviations. Averaging alone
is slightly below the 64000-trial reference (-0.000971 in the old cohort), so
it cannot establish final headroom. A better calibration variant must still
demonstrate a positive paired confidence bound on the hidden test.

## Scientific result before this iteration

REVIEW.md's reference scores were 0.9997 / 0.9992 / 0.9741 / 0.9312
(main / wide / extreme / restricted capacity). Main and wide were effectively
saturated; cherry-picking reservoir seeds paid about 0.03–0.04, comparable to
honest improvements. The earlier REDTEAM.md reported learned timing at
0.996 / 1.000 against reference 0.960 / 0.674, but enumeration of 31 timings
reached 1.000 / 1.000. Constant forward tied that reference. These were toy
kinematic results, not evidence of learned control of a robot.

The replacement MuJoCo audit produced these held-out results before the latest
combined changes (extreme / restricted capacity / noisy observations):

| Controller | Success rates |
|---|---|
| Eq. 1 reference, 2000 trials | 0.53442 / 0.48931 / 0.21348 |
| Eq. 1, 8000 trials | 0.92188 / 0.92441 / 0.65393 |
| Averaged late readout | 0.85356 / 0.90032 / 0.40498 |
| Rule imitation | 0.99993 / 0.98975 / 0.75591 |
| Constant forward | 0.20776 / 0.20166 / 0.20044 |

That setup is rejected: the hand rule still pays. Its aggregate scores are in
`redteam/final/rejected_robot_setup/`. The job was cancelled after these complete
comparisons; uncompleted methods are not assigned invented results.

## Changes fixed before the next combined job

The simulator now has a torque-limited three-joint MuJoCo robot, inverse
kinematics, cubic task-space primitives, joint inertia and contact dynamics at
1 kHz. Geometry and software neurons are documented substitutes for the
unavailable authors' robot and silicon. The paper's six observations, two
primitives, Gaussian input spikes, fixed random feedforward reservoir, local
Eq. 1 update and two-trial update interval are retained. Reward shaping now
uses instantaneous distance, as Eq. 2 specifies. Evaluation samples softmax
actions, as in the paper, with identical action draws for paired comparisons.

The robot audit showed unfinished learning at 8000 and 32000 trials. The reference now
uses 64000 trials rather than the paper's 2000; the learning rule, reward,
learning rate and discount remain unchanged. This departure is explicit and
is not an attempt to match a published number. The final checkpoints come from
the same exact Eq. 1 trajectories already run in the combined training job.
The primary honest alternative chooses one global readout blend, temperature
and action bias per experiment on 256 fresh public validation launches per
chip, with identical validation inputs across candidates. Its candidate set is
fixed before hidden scoring. Fixed 8x sharpening and averaging updates after
trial 32000 are secondary alternatives. The 32000- and 2000-trial checkpoints
and an untrained readout are weaker controls. Calibration never imports the
hidden evaluator or uses test outcomes to choose a readout.

The two old saturated conditions carry no rank. Extreme speeds carry weight
0.2, restricted capacity 0.3 and sensing robustness 0.5. Robustness now uses
1020 units, launch x in [0.80, 1.35] m, incoming headings up to 0.30 rad,
0.035 m position noise, 0.35 m/s velocity noise and 40 ms observation latency.
This extends the paper's sensing-uncertainty discussion; it is not a reported
paper experiment. Restoring the full 1020-unit scale here avoids attributing
arbitrarily restricted capacity to the paper's method. Stronger sensing
uncertainty requires temporal control rather than a fixed instantaneous
intercept schedule. Whether this succeeds is an empirical check, still pending.

All ten chip instances are frozen per slot. Each controller sees the same 4096
launches, observation noise and action draws. The audit reports paired gains,
chip and episode standard errors, and the earlier conservative normal bounds.
The final headroom check additionally uses 4096 paired bootstrap replicates:
it jointly resamples chip identities and whole episode columns, sharing every
resampling draw across methods and chip identities across conditions. This
estimates uncertainty for bounded, skewed gains without assuming Gaussian chip
effects. Cohort normalization is fixed at its measured value. The primary
public-calibrated variant must have a positive 95% lower bound; secondary
variants cannot substitute for a failed primary check. The audit also
checks a fixed 27-member bank of intercept schedules crossed with temporal
smoothing constants, and public-data rule imitation with sharpened logits.
It must show a stronger variant above reference beyond paired noise and both
rule comparisons below reference before finalization. No bound or headroom
claim is accepted merely because a number is less than one. The final reference
is below the success bound in all three scored experiments (0.99697, 0.99092,
0.84060). The added sensing experiment provides substantial remaining error;
neither the timing bank nor the learned readouts tested so far solve it.

## Shortcut repairs and verification status

| Major shortcut and previous outcome | Repair | Final score |
|---|---|---|
| Forged summaries: 1 / 1; impossible rates: 2 / 2 | Ignore submitted metrics; execute raw bounded numeric controllers and recompute outcomes | 0 / 0 / 0 in completed audit; final-reference replay pending |
| Failed task compensated by huge auxiliary numbers: rank +0.387 | Rank only task success; capacity is a hard constraint, other metrics diagnostic | Forged metrics receive complete worst values; final rank pending |
| Empty or missing experiment: free tie | Discover every method on disk; require all experiments and ten controllers; failure zeros every experiment | 0 / 0 / 0 in completed audit; replay pending |
| Omitted or fictitious capacity: reference tie | Count actual neurons and compare every fixed chip array to evaluator-owned assets | 0 / 0 / 0 in completed audit; replay pending |
| Seed selection: +0.03–0.04 | Fixed chip identities; even copying seed 0 with tiny readout jitter fails | 0 / 0 / 0 in completed audit; replay pending |
| Known-seed lookup and 31-way search: 1 / 1 | Fresh hidden inputs; only readout arrays execute, with no submitted planner or simulator access | Interface closed; rerun affected controls |
| Replacing the public simulator: 1 / 1 | Test owns independent simulator, robot, runtime and chip copies | Interface closed |
| Constant action: old reference tie | Physical robot and broad launches | 0.20776 / 0.20166 / 0.19458; final rank pending |
| Hand-written timing: nearly perfect; imitation beats robot reference | Stronger sensing task, full paper capacity, converged training budget; audit filtered rules and imitation | Bank 1 / 1 / 0.72583; imitation 0.99954 / 0.95159 / 0.69504; final cohort pending |

Wrong capacity, nonfinite weights, missing experiment, forged JSON, copied chips
and constant forward are replayed through the same hidden interface. Invalid
runs get complete worst values (zero success/contact, units at budget) in every
experiment. Unknown method directories are not omitted by a training allowlist.
Only per-experiment `results` in `scores_train.json` and `scores_test.json` are
ranked. Direction comes from `higher_is_better`, coefficients use absolute
magnitude, and improvements are normalized by across-attempt standard deviation.

## Isolation, reproducibility and packaging

Test instance generation and all raw paired outcomes stay inside `eval/test/`.
The test never trains and never imports method code. It executes supplied
readout arrays on its own frozen assets, so a private simulator cannot search
the hidden instance. Every scored value comes from those physical rollouts.
No wall-clock metric or method time limit affects the reference. All learning
finishes before serial reference and baseline scoring; final score files contain
reference runs and evaluator-owned baselines alone. Original reported paper
numbers remain unchanged under `_paper_reported` in `scoring/reference.json`.

An early job was cancelled for unstable MuJoCo accelerations. Explicit contact
stiffness/damping replaces an unresolved time constant; finite resets, stalled
rollouts and numerical explosions now fail before scoring. All affected
controllers and calibrations were discarded. A public physics preflight runs
before the next training phase.

Worker drains also repeatedly restarted scoring. Audit-only raw checkpoints
are keyed by complete controllers, inputs, sources and library versions; normal
no-argument evaluators remain uncached. Changed evaluator code invalidates all
old raw outcomes. Training checkpoints save exact weights, averaging state,
online history and RNG states between two-trial updates. A compute-side check
requires interrupted and uninterrupted learning to be bit-identical. No
elapsed-time stopping rule is introduced.

Overlapping worker retries exposed a shared temporary-score filename in an
earlier job. Both independent evaluators, raw caches and audit metadata now use
unique temporary files and atomic replacement. A compute-side concurrent-writer
check covers the repair. Exact full-source and pinned-runtime migration digests
preserve existing raw rollouts only across that persistence-only change; normal
evaluators remain uncached. The 64000-trial reference promotion changes no
physics, inference, training trajectory, or rollout-cache compatibility key.

Local source parsing and shell checks pass. The currently stored score files
are stale/incomplete audit artifacts, and `validate.py` correctly rejects them.
The combined job regenerates all final score files before running validation.
Final measured scores, uncertainty, headroom bounds and the definitive shortcut
before/after table still need to replace this draft.

## Remaining scope

Provenance and selection among repeated training runs on the same assigned
chip are left to the separate attempt grader. The evaluator checks chip identity,
not an unverifiable learning history. Alternative optimizers, longer training,
public-data search and imitation remain expressible learning procedures. Their
ability to query a hidden instance during evaluation is closed. A short rule
that still outranks reference would remain a test flaw, not a leftover to
justify with that grader. This gym does not measure electrical power, exact
silicon performance or exact fidelity to the authors' robot.
