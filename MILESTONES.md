# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 16:19] - core_claim
- Existing job `47cdad61-a3f` reproduced the central randomized air-hockey claim with 0.960 success using the spiking reservoir/e-prop controller and 1020 hidden neurons.
- Existing job `b0836d70-c9d` added the wide velocity-range setting, but the reproduced score remains below the paper's reported value, so the milestone stays at `core_claim`.

### [2026-06-02 16:22] - core_claim
- Continuation audit confirmed the implementation remains an identifiable software reproduction of the paper's spiking reservoir plus e-prop readout update, and `scripts/evaluate.sh` remains independent from `method/`.
- Removed an unscored planned target from the scoring files and reusable run scripts so every target listed in `TARGETS.md` has a corresponding entry in `scores.json`.

### [2026-06-02 16:43] - core_claim
- Fresh continuation audit restored the milestone from `method_runs` to `core_claim` after checking the paper, method code, output artifacts, evaluator independence, and deliverables.
- Removed leftover briefing template text and corrected `scripts/evaluate.sh` usage comments; validation still passes on the accepted score file.

### [2026-06-02 16:58] - core_claim
- Updated the default evaluator so `scores.json` records both the random sanity baseline and the spiking e-prop controller for each scored experiment.
- Updated `scripts/run.sh` to verify `scores.json`, the train/test split scores, and `validate.py --compare` in one CPU job.

### [2026-06-02 17:03] - core_claim
- Final verification confirmed the `core_claim` milestone remains justified.
- Cleaned method-agnostic documentation and scoring terminology by using `controller_units` for the compact-controller constraint, then reran `bash scripts/run.sh` successfully.

### [2026-06-02 17:06] - core_claim
- Prepared an incremental CPU job for the paper's narrow velocity-range condition `[0.7, 0.9] m/s`, which is a candidate for `core_claim_plus` if the reproduced method clearly beats the random sanity baseline.
- Extended the artifact evaluators to recognize `speed_narrow` as `velocity_range_narrow` and to carry `train_success_gain` into the default `scores.json` when method artifacts provide training summaries.

### [2026-06-02 17:10] - core_claim
- Final verification removed the aborted narrow-condition submission setup from the packaged scripts because that experiment was not accepted into the scored gym surface.
- Restored `scripts/run.sh` to validate the existing artifact scores, train/test score files, and `validate.py --compare`; the milestone remains `core_claim`.

### [2026-06-02 17:29] - core_claim
- Re-ran final verification checks; `scripts/run.sh` and `validate.py --compare` pass over the packaged artifacts.
- Added explicit notes that random-policy reference values are local sanity baselines and removed method-family wording from the high-level research-direction guidance.

### [2026-11-12] - core_claim (hardening)
- Rebuilt the task in the paper's geometry: lateral puck launch, spline primitives, 1 ms physics, 100-step episodes.
- Submissions are now controller configurations, which the evaluators execute on held-out episodes. Every number is recomputed; failures get worst values.
- Restored Eq. 1 exactly, 1 ms spiking dynamics, feedforward random synapses and 10 reservoir seeds. Learning rate 0.012 was chosen on the visible slice.
- Reference held-out success: 0.9997 / 0.9992 / 0.974 / 0.931 (main / wide [0.7,1.5] / extreme [0.5,2.0] / 256-unit extreme); random 0.10, always-forward 0.
- Shortcut outcomes, leftovers and the headroom analysis are in `HARDENING.md`.


### [2026-11-17] - method_runs
- The earlier milestone entries describe superseded toy dynamics. Robot audit
  results show the 2000-trial controller below a timing rule, so final scientific
  hardening is not yet demonstrated. The new combined audit must establish it.
- Eq. 1 runs on the software spiking substrate and torque-limited robot.
  Extended training and the sensing-robustness setting remain under evaluation.

### [2026-11-18] - method_runs
- All 30 extended-training trajectories completed; saved held-out comparisons
  show improvement over the reference and failure of malformed submissions.
- Certification still awaits the fixed timing-rule bank, paired uncertainty
  and isolated canonical scores. A temporary score-file collision is repaired
  without changing the scientific setup or discarding valid raw rollouts.

### [2026-11-18] - method_runs
- The completed 32000-trial audit failed certification: a short timing bank
  narrowly outranked reference, and skewed paired gains defeated its normal
  uncertainty check. The worker was then lost during canonical scoring.
- The completed exact 64000-trial checkpoints become the final reference.
  Public-only readout calibration and paired chip/episode resampling are queued
  together with final variant/attack comparisons and canonical scoring.
  The milestone remains `method_runs` until headroom is demonstrated.


### [2026-11-18] - core_claim
- Job `516de1db-87c` passed the final scientific audit and validation. Exact
  64000-trial Eq. 1 reference success is 0.996973 / 0.990918 / 0.840601 under
  the 1020 / 256 / 1020 unit constraints on the final software robot.
- Public-only readout calibration scores 0.999756 / 0.998730 / 0.864893. Its
  paired rank gain is +0.040983, 95% bootstrap interval [0.022084, 0.062891],
  positive on all ten chips. Shorter training and simple timing/constant
  controls score below reference. All five malformed attacks fail every task.
- Canonical reference/baseline train and test files were regenerated in
  isolation. The milestone covers learned compact robotic control in the
  documented software substitute, not silicon fidelity or the paper's
  2000-trial sample efficiency. Final evidence and limitations: HARDENING.md.


### [2026-11-21] - core_claim (final hardening: learning budget)
- The scored quantity is now the paper's: online success within 2000 trials per
  chip, on 20 hidden chips and the paper's own three launch conditions; the
  evaluator runs each submitted learner online and owns every outcome.
- Reference (Eq. 1 exactly, learning rate 0.012 chosen on public chips):
  final success 0.846 / 0.751 / 0.585 and online success 0.680 / 0.594 / 0.421
  (randomized / wide / extreme), versus random 0.07; success falls with wider
  speed range, as in the paper (paper: 0.98 / 0.93 / 0.86 on silicon).
- Headroom: a standardized-input e-prop learner chosen on public chips gains
  +0.586 paired weighted success (CI [0.483, 0.701], 20/20 chips). All weaker
  variants and shortcuts score lower beyond noise. Details: HARDENING.md.
