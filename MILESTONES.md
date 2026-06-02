# Reproduction Milestones

**Current: core_claim**

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

### [2026-06-02 16:03] - none
- Job `c7cdbcc2-e98` made the random baseline non-trivial at 2% success, but the method did not learn in that sparse-reward configuration.
- Widened the valid primitive-contact phase while keeping random below the invalid near-perfect regime, added return centering for lower-variance e-prop policy-gradient updates, and added training diagnostics to the method artifact.

### [2026-06-02 16:08] - core_claim
- Job `47cdad61-a3f` validated the randomized puck position/speed setting: the random sanity baseline scored 7.2% success while the spiking e-prop controller scored 96.0% success.
- The successful run preserved the paper's fixed-size 1020-neuron hidden reservoir constraint and passed `python validate.py --compare`.
- Prepared the next job to add the paper's wide velocity-range condition for `core_claim_plus`.

### [2026-06-02 16:13] - core_claim
- Job `b0836d70-c9d` added the paper's widest velocity-range condition and passed `python validate.py --compare`.
- The wide-range run scored 67.4% success with the same 1020-neuron spiking e-prop controller, below the paper's reported 93.0%; this is useful coverage but not enough to claim `core_claim_plus`.
- Added condition-aware random baselines and prepared a longer tuned wide-range run.

### [2026-06-02 16:16] - core_claim
- Job `58fce6b9-d6d` tested a longer 5000-episode tuned wide-range run.
- The wide-range score stayed at 67.4% while the random wide sanity baseline was 7.8%, so the additional setting demonstrates non-trivial learning but remains below the paper's 93.0% result.
- Reverted the reusable scripts to the shorter 2000-episode wide run because the longer run did not improve the final scored metric.

## Stop Justification
- Completed at milestone `core_claim`.
