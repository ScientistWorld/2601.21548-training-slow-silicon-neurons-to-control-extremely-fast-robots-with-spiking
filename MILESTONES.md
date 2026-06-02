# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 00:00] - method_runs
- Implemented and ran the paper's spiking e-prop reinforcement-learning controller end-to-end on the air-hockey task-level simulator.
- Verified the pipeline writes method artifacts and `scoring/scores.json` through the method-independent evaluator.

### [2026-06-02 00:00] - core_claim
- Reproduced the central randomized puck-position/speed claim on the accessible simulator: `spiking_eprop` reached 0.960 success and 0.960 hit rate with 1020 hidden neurons.
- Confirmed the task is non-trivial by running a random-action sanity baseline that remained far below the learned controller in prior job artifacts.
- Added a second scored wide-velocity condition; it remains below the paper target and is documented as future work rather than claimed as `core_claim_plus`.

### [2026-06-02 16:14] - core_claim
- Continuing-agent audit found the method implementation faithful to the paper's computational structure rather than a surrogate.
- Re-ran `bash scripts/evaluate.sh` and `python validate.py --compare`; scoring, standalone evaluation, and import-separation checks all passed.
- Restored the reset milestone to `core_claim`.

## Stop Justification
- Completed at milestone `core_claim`. Higher milestones would require improving or expanding the wide velocity-range reproduction.
