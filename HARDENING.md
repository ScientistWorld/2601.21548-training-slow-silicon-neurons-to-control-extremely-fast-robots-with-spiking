# Hardening report

**Verdict.** Better results now score higher, weaker ones lower, and every major shortcut from `REDTEAM.md` now scores as a failed run. Three things are left open:
- Seed cherry-picking still pays (+0.03 to +0.04 on the hard experiments).
- Distilling a planner or hand-coding the intercept rule cannot be detected by the test.
- The paper's sample-efficiency claim is not scored.

All evidence comes from three CPU jobs (`jobs/ad7ac52b-6c0`, `jobs/7556e429-9c0`, `jobs/ffe3c17e-72b`). Results are under `redteam/hardening/results/` and the driver is `redteam/hardening/run_hardening.py`. Each case ran through the unmodified `bash scripts/evaluate_test.sh` with no arguments.

## Why the gym was rebuilt rather than patched

The red-team audit showed three linked failures:
1. **The evaluator trusted the method.** It copied `success_rate` and the other numbers from JSON the method wrote.
2. **The test held no inputs of its own.** Evaluation seeds were visible in `method/train.py`.
3. **The task did not need timing.** The puck came head-on, was advanced twice per step, and contact was gated by swing age. The trained reference was a constant "forward" policy, and 31-delay enumeration solved every episode.

Fixing only the evaluator would still have left a task in which constant behaviour scores 0.94. I therefore restored the paper's setup and changed what is submitted.

### Task, now in the paper's geometry (`data/air_hockey_env.py`)
- 1.948 x 1.038 m table, puck radius 0.03165 m and mallet radius 0.04815 m (Fig. 1).
- The puck is launched **laterally** from the table edge in the generation band x = 0.95-1.05 m, and bounces off the walls.
- Home and forward primitives target 0.70 m and 1.50 m (Table I; Fig. 1 labels the forward point 1.4 m).
- Primitives run as open-loop cubic splines from the current end-effector state; corrective maneuvers are compositions of primitives.
- 50 Hz control and 1 ms physics, with Eq. 2 reward and success.
- Episodes last up to 100 steps; Fig. 2b shows 100-step episodes before training.

On this task, always-forward scores 0, random about 0.10, and the best fixed wait 0.82 / 0.53 / 0.37 in a pure-Python prototype.

### Submissions are controllers, executed by the evaluator
- A method submits 10 controller configurations per experiment in the format of `data/neuromorphic_chip.py`. That runtime is a software stand-in for DYNAP-SE with its FPGA spike generators and host readout.
- `eval/test/` runs each controller in closed loop on 1000 held-out episodes. Its episode seed exists only there, and it uses frozen copies of the simulator and runtime.
- Every metric comes from those rollouts. Hidden units are counted from the file.
- Per-episode search cannot run at decision time, because the controller is a network the evaluator executes.

### The paper's method is restored where the earlier iteration had simplified it (`method/`)

| Element | Before | Now |
|---|---|---|
| Readout update | Within-episode standardised returns, length normalisation, probability floor | Eq. 1 exactly, every two trials |
| Hidden neuron dynamics | Sigmoid-Bernoulli spikes once per 20 ms | LIF with synaptic filter, adaptation and refractoriness at 1 ms |
| Input spikes | Poisson counts | Deterministic spike generators |
| Connectivity | Random input synapses plus sparse recurrence | Fixed random feedforward input synapses only (Sec. II-B) |
| Seeds | 1 | 10 reservoir seeds per experiment (Sec. II-D) |
| Training episodes | 2000 / 5000 | 2000 |

The learning rate is not given in the paper. It was selected on the **visible slice only** (`results/select_lr.json`), among 0.007, 0.012 and 0.025 (scaled by readout input power):

| Learning rate | Visible-slice score | Behaviour |
|---|---|---|
| 0.025 | 0.96 | Originally picked by a one-seed sweep; collapsed irreversibly to zero success in 2 of 15 runs |
| 0.007 | 1.21 | Learned too slowly on the extreme range |
| **0.012** | **1.27** | Selected |

## Requirement checklist

| Requirement | Status |
|---|---|
| Test inputs only in `eval/test/` | Held-out episodes are drawn from a seed in `eval/test/evaluate.py`; simulator and runtime are frozen copies there |
| Every number recomputed from raw outputs | Success, hit, worst-seed and units come from rollouts and the configuration file; nothing the method writes is read |
| Comparison fixed by `eval/` | Episodes, test size (10 x 1000 per experiment), unit budgets, baselines (`random_policy`, `always_forward`) and action rule (argmax) |
| Impossible values / failed constraints fail | Non-finite, out-of-bounds, over-budget, malformed, missing or duplicated controllers give the worst value in every metric (success 0, units = budget) |
| Missing entries not free | Every proposed method in `reference.json` is always scored, so an empty submission scores 0 |
| `evaluate_test.sh` with no arguments scores every method on disk | Yes, in parallel over CPUs; never crashes on a submission; deletes stale output first |
| Train/test parity | Separate code paths, same protocol. Reference visible vs held-out: 0.9998/0.9997, 0.9996/0.9992, 0.977/0.974, 0.936/0.931 |

## Do better results score higher?

All scores below are on the held-out test with 10 seeds. The rank is the specified rule (|coefficient| x weight x directed improvement / population standard deviation across the scored cohort), computed in `redteam/hardening/analyze.py`. Its magnitudes are illustrative, because the future cohort is unknown. Seed noise is the standard error of the difference in per-controller success: paired for snapshots of the same runs, unpaired otherwise.

| Method | main | wide | extreme | single-core | Rank vs reference |
|---|---:|---:|---:|---:|---:|
| **Reference `spiking_eprop`** (Eq. 1, 2000 episodes) | 0.9997 | 0.9992 | 0.974 | 0.931 | 0 |
| More training: same runs at 4000 episodes | 1.000 | 1.000 | 0.987 (z=0.8) | 0.966 (z=1.8) | +0.026 |
| Conjunctive reservoir: Eq. 1 with hidden units as coincidence detectors over (y_p, v_y) | 0.940 | 0.933 | 0.825 | **0.985 (+0.053, z=3.1)** | -0.225 |
| Composed: 4000 episodes on the 1020-unit experiments + conjunctive on single-core | 1.000 | 1.000 | 0.987 | 0.985 | **+0.057** |
| Actor-critic e-prop (not Eq. 1) | 1.000 | 0.900 | 0.883 | 0.776 | -0.709 |
| Planner distillation (out of bounds) | 0.820 | 0.824 | 0.824 | 0.792 | -0.578 |
| Weaker: 500 episodes (same runs) | 0.882 | 0.631 | 0.401 | 0.432 | -1.625 |
| Weaker: untrained readout (same runs) | 0.143 | 0.141 | 0.105 | 0.110 | -3.157 |
| Baseline `random_policy` | 0.098 | 0.101 | 0.098 | 0.100 | -2.916 |
| Baseline `always_forward` | 0 | 0 | 0 | 0 | -3.274 |

Notes on the table:
- The critic variant collapses on some seeds; its worst-seed success is 0.
- The composed row is computed rather than run separately: the rank sums per-experiment contributions, and each experiment's controllers are scored independently on the same episodes.

**What this shows:**
- **Ordering:** more training, the critic, and the weaker variants (untrained, 500 episodes) order as expected. The weaker variants sit far below the reference, beyond noise (z from -1.9 to -16).
- **A clear better result exists:** the conjunctive representation gains +0.053 on the 256-unit experiment at 3.1 standard errors.
- **Gains within noise:** more training gains +0.013 and +0.035, but with 10 seeds that is only 0.8-1.8 standard errors.

From earlier evidence, with 5 seeds:
- Finer 20-cell encoding scored lower (0.945 / 0.830 on extreme / single-core).
- The rejected learning rates scored 0.824 / 0.903 (0.007) and 0.772 / 0.666 (0.025).

**Before, from `REDTEAM.md` (old task, two experiments):**
- Reference: 0.960 / 0.674.
- A learned delay table reached 0.996 / 1.000 (+0.255).
- No learning: 0.597 / 0.492.
- 10,000 episodes: unchanged.

In that gym the direction was right, but a constant action tied the reference.

## Shortcuts: fix, score before and after

"After" is the held-out test with 10 seeds (`results/shortcuts.json`; the 5-seed run `shortcuts_n5.json` gives identical outcomes). Every case exited 0.

| Shortcut (REDTEAM) | Before | Fix | After |
|---|---|---|---|
| Fabricated metric JSON | 1 / 1, +0.263 | Numbers are never read; the proposed method without controllers is scored | 0 everywhere, -3.52 |
| Impossible values (2.0, -1020 units) | 2 / 2, +2.578 | Same | 0, -3.52 |
| Non-finite values | NaN accepted | Configuration validation (finite, bounded) | 0, -3.52 |
| Omit / understate capacity | Tie | Units counted from the file; per-experiment budget | Over-budget (2040 units, or 512 in single-core): 0, -3.52 |
| Empty submission | Tie | Proposed methods always scored | 0, -3.52 |
| Missing experiment | Tie | Worst values per missing experiment | 1.000 / 0 / 0 / 0, -2.21 |
| Unlisted extra method skipped | Tie | Every directory under `outputs/` is scored | Extra method scored (0) |
| Replace public simulator (relaxed gate) | 1 / 1, +0.262 | Test uses frozen copies | 0 on test for the edited-simulator controller |
| Move baseline | Tie | Baselines computed inside the evaluator | Unaffected |
| Constant forward | 0.960 / 0.674 (tie/win) | Lateral task needs timing | 0, -3.28 |
| Brute-force delay enumeration | 1 / 1, +0.262 | Controllers are networks executed by the evaluator; no per-episode compute | Not submittable. Enumeration with a simulator copy would solve 100% of held-out episodes (`private_search.json`); the distilled version scores 0.79-0.82, below the reference |
| Seed lookup of evaluation streams | 1 / 1 | Held-out episodes drawn in `eval/test/` only | Not possible |
| Failed task, forged auxiliary metrics | +0.387 | Numbers never read | 0, -3.52 |
| Stale scores after a failing artifact | Exit 1, stale file kept | Old file deleted first; bad artifacts are caught and scored as failures | Corrupt file with planted stale perfect scores: exit 0, 0 |
| *New:* duplicated controller across seeds | - | Identical configurations rejected | 0 |
| *New:* full-size controller in the 256-unit experiment | - | Per-experiment budget | 0 on single-core |
| *New:* out-of-bounds neuron parameters | - | Bounds check | 0 |

## Shortcuts left open, and why

1. **Seed cherry-picking.** A method can train extra runs and submit the best 10 by visible-slice success. With 5 required controllers, picking the best 5 of 10 gave 1.000 / 0.9998 / 0.997 / 0.971 against the reference's 1.000 / 0.9992 / 0.972 / 0.929. That is +0.026 and +0.042 on the hard experiments, comparable to a genuine improvement. The evaluator cannot see discarded runs, and only an evaluator-run training could close this, which the light hidden test excludes. Requiring 10 controllers (the paper's count) and rejecting identical ones narrows the crudest form. `CONSTRAINTS.md` and `DIRECTION.md` forbid it, and the grader must check it, e.g. against the shipped `training_seed_<k>.json` logs.
2. **Planner distillation or a hand-written intercept rule.** In this substitute, launching when |y_p| < a + b|v_y| would succeed almost always in all four settings. A controller that encodes it, or that imitates per-episode simulated plans, is a valid configuration. The simple distillation tried here (behaviour cloning on the reference reservoir) scored 0.79-0.82, below the reference. A better-executed one, or a hand-designed network, could exceed it. Both are out of bounds in `DIRECTION.md`. The test cannot tell them from learned controllers, because the substitute physics is simple enough to write down. Moving to the paper's MuJoCo robot would raise that bar but not remove it.
3. **Training budget.** The paper's sample-efficiency claim (stable within about 1300-1500 episodes) is not scored. Training budget cannot be verified without retraining, and more training legitimately helps a little (+0.013 and +0.035).
4. **Small capacity reductions.** `controller_units` has coefficient 0.1. Dropping a few units is a real but tiny compactness gain. Variance normalisation could amplify it if nearly all attempts sit at the budget. I left it because it rewards the paper's compactness claim and cannot outweigh a task-success loss.

## Reference score before and after

| | main | wide | extreme | single-core |
|---|---:|---:|---:|---:|
| Before: old task, held-out (copied numbers) | 0.960 | 0.674 | - | - |
| Intermediate: restored method, learning rate 0.025, 5 seeds (seed collapse) | 0.996 | 0.790 | 0.772 | 0.666 |
| **After: final reference, 10 seeds, held-out** | **0.9997** | **0.9992** | **0.974** | **0.931** |
| Paper (online success) | >0.98 | 0.93 | 0.86 | - |

The paper's direction is reproduced: success falls as the speed range widens, and falls further with fewer neurons. The substitute is easier than the robot at [0.7, 1.5] m/s.

## Headroom

- **At the bound:** the reference sits at the bound of success, worst-seed success and hit rate on `randomized_air_hockey` and `velocity_range_wide` (0.9997, 0.9992). Those two experiments now measure reproduction.
- **Below the bound:** `velocity_range_extreme` is at 0.974 (worst seed 0.907), and `controller_units` sits at the budget everywhere.
- **Added `single_core_extreme`:** the paper's widest range with 256 units (one DYNAP-SE core), where the paper's own limitation, the finite resolution of a fixed-size network, binds. The reference scores 0.931 (worst seed 0.845), random 0.100, always-forward 0.
- **Variants above the reference:** more training (0.966) and the conjunctive representation (0.985) score above it. The search that the declarative interface rules out would reach 1.000.
- **Weights:** changed from 0.75 / 0.25 to 0.4 / 0.2 / 0.2 / 0.2. The headline setting is saturated, and the paper's claims about range and capacity are where methods differ. Main keeps the largest single weight as the paper's headline result.

## What I could not fix

- **The environment is still a substitute:** a task-level 2D simulator with kinematic spline primitives, not the MuJoCo robot with inverse kinematics.
- **The hardware is software:** LIF with adaptation, not the exponential AdEx term or analog noise.
- **Different success measure:** the gym scores final greedy controllers, while the paper reports online sampled success.
- **One clear gain only:** the only demonstrated gain beyond seed noise is on `single_core_extreme`. On `velocity_range_extreme` the best honest variant is within noise (z=0.8).
- **Hidden test cost:** about 4 minutes on 8 CPUs (`eval_reference.json`). On a single CPU it scales to roughly 15-20 minutes for 10 x 1000 rollouts per experiment.
- **Validator warnings:** `validate.py` prints soft "eval looks detached" warnings, a regex heuristic. The evaluators do read `outputs/<method>/...` through path objects. All checks pass.
- **Superseded tooling:** the old audit scripts under `redteam/` (outside `redteam/hardening/`) target the superseded JSON interface. They are kept as evidence of the "before" state and do not run against the new evaluator.
