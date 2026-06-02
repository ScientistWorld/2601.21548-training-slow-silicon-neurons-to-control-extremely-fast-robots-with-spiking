# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 16:19] - core_claim
- Existing job `47cdad61-a3f` reproduced the central randomized air-hockey claim with 0.960 success using the spiking reservoir/e-prop controller and 1020 hidden neurons.
- Current iteration adds the paper's fixed-position variable-speed condition to scoring and submits a CPU job to test whether the reproduction can advance to `core_claim_plus`.

### [2026-06-02 16:22] - core_claim
- Continuation audit confirmed the implementation remains an identifiable software reproduction of the paper's spiking reservoir plus e-prop readout update, and `scripts/evaluate.sh` remains independent from `method/`.
- Found that the newly listed `variable_speed_generalization` target is not yet present in `scores.json`; resubmitting the prepared CPU job to generate that missing score.
