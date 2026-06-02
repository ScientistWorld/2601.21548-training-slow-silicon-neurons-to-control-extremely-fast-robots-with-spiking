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
