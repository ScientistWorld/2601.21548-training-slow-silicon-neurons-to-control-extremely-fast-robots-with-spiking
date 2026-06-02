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

### [2026-06-02 15:59] - none
- Job `7595de2e-7e4` ran to completion and produced a high method success rate, but the random sanity baseline also achieved 100% success.
- Tightened the task simulator so successful contact requires an active forward mallet swing, matching the paper's emphasis on precise timing and preventing trivial intercept-point camping.

### [2026-06-02 16:01] - none
- Job `53701b5a-4a7` ran to completion, but random action selection still reached 97.3% success while the method reached 95.3%.
- Updated the simulator to treat actions as persistent open-loop primitives with a narrow valid swing-contact phase, matching the paper's ballistic primitive description.

## Stop Justification
- Not stopped. First reproduction job is ready for submission.
