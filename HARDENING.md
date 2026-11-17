# Final hardening (audit pending)

The reports identified paying shortcuts and inadequate headroom. Final
verification is still pending: the first robot audit also failed the timing-rule
check. Its results are preserved, not presented as a successful hardening.
The revised 32000/64000-trial audit was interrupted by worker-node loss after
16 of 30 training runs completed. Exact training/RNG checkpoints retain partial
progress; the continuation changes audit bookkeeping only. Final hidden scores
and paired headroom checks remain pending.

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

The robot audit showed unfinished learning at 8000 trials. The reference now
uses 32000 trials rather than the paper's 2000; the learning rule, reward,
learning rate and discount remain unchanged. This departure is explicit and
is not an attempt to match a published number. The honest variants are a
64000-trial continuation and averaging updates after trial 32000. A 2000-trial
checkpoint and an untrained readout are weaker controls. No hidden result
selects their hyperparameters or training trajectory.

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
chip and episode standard errors, and a conservative uncertainty bound. It also
checks a fixed 27-member bank of intercept schedules crossed with temporal
smoothing constants, and public-data rule imitation with sharpened logits.
It must show a stronger variant above reference beyond paired noise and both
rule comparisons below reference before finalization. No bound or headroom
claim is accepted merely because a number is less than one.

## Shortcut repairs and verification status

| Major shortcut and previous outcome | Repair | Final score |
|---|---|---|
| Forged summaries: 1 / 1; impossible rates: 2 / 2 | Ignore submitted metrics; execute raw bounded numeric controllers and recompute outcomes | Pending rerun |
| Failed task compensated by huge auxiliary numbers: rank +0.387 | Rank only task success; capacity is a hard constraint, other metrics diagnostic | Pending rerun |
| Empty or missing experiment: free tie | Discover every method on disk; require all experiments and ten controllers; failure zeros every experiment | Pending rerun |
| Omitted or fictitious capacity: reference tie | Count actual neurons and compare every fixed chip array to evaluator-owned assets | Pending rerun |
| Seed selection: +0.03–0.04 | Fixed chip identities; even copying seed 0 with tiny readout jitter fails | Pending rerun |
| Known-seed lookup and 31-way search: 1 / 1 | Fresh hidden inputs; only readout arrays execute, with no submitted planner or simulator access | Interface closed; rerun affected controls |
| Replacing the public simulator: 1 / 1 | Test owns independent simulator, robot, runtime and chip copies | Interface closed |
| Constant action: old reference tie | Physical robot and broad launches; measured 0.20–0.21 in rejected robot setup | Pending final rerun |
| Hand-written timing: nearly perfect; imitation beats robot reference | Stronger sensing task, full paper capacity, converged training budget; audit filtered rules and imitation | Pending final comparison |

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
