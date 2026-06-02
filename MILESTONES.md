# Reproduction Milestones

**Current: none**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-02 15:54] - none
- Read the paper markdown and extracted the core method, task conditions, reward, and reported numbers.
- Implemented the task-level air-hockey simulator, spiking reservoir/e-prop readout controller, random sanity baseline, and method-agnostic evaluator.
- Prepared the first compute job to run the reproduced algorithm and score the randomized puck position/speed experiment.

### [2026-06-02 15:57] - none
- Job `f65b0a96-f82` failed before running the method because direct script execution could not import workspace packages.
- Updated all shell entry points to export `PYTHONPATH=/home/user` before invoking Python.

## Stop Justification
- Not stopped. First reproduction job is ready for submission.
