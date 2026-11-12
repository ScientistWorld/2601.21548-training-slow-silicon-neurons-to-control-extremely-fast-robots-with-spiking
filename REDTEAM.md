> **Superseded.** This document describes the gym before the final hardening iteration (head-on simulator, self-reported JSON scores, two experiments). The task, submission interface, evaluators and scoring changed; see `HARDENING.md`.

# Does this gym reward better research?

Honest improvements score higher on the substitute task, and the no-learning ablation scores lower, both beyond the measured seed spread. **The gym nevertheless fails to distinguish scientific contribution from shortcuts.** The substitute task does not establish the paper's learned robotic timing effect: every trained reference controller chooses the same constant forward action, and enumerating just 31 waiting times solves every evaluated episode. The test is too small and too trusting to support the paper's claim.

## Better and weaker methods, tested first

All numbers below come from the original hidden test's `results`, after running `bash scripts/evaluate_test.sh` with no arguments. Main means `randomized_air_hockey`; wide means `velocity_range_wide`. Success and hit rates coincide for these controls. Each trained control uses 1020 declared units, 500 evaluation episodes per seed, and the original simulator.

| Method and saved test scores | Main success | Wide success | Rank change vs packaged reference |
|---|---:|---:|---:|
| [Packaged reference, rerun seed 0](redteam/quality_reference_seed0/scores_test.json) | 0.960000 | 0.674000 | 0 |
| [Matched reference, seeds 0–2](redteam/quality_reference_three_seeds/scores_test.json) | 0.944000 | 0.666000 | −0.032278 |
| [Weaker: no learning updates](redteam/quality_weak_no_learning/scores_test.json) | 0.597333 | 0.492000 | −0.732019 |
| [More training: 10,000 episodes per seed](redteam/quality_more_training/scores_test.json) | 0.944000 | 0.666000 | −0.032278 |
| [Better: learned timing, seeds 0–2](redteam/quality_learned_timing/scores_test.json) | 0.996000 | 1.000000 | +0.255275 |
| [Simple random-action baseline](redteam/quality_random_baseline/scores_test.json), zero adaptive units | 0.071333 | 0.079333 | −1.880903 |

The better controller genuinely learns reward values for 21 waiting times in eight observed-speed contexts from 2000 online episodes. It receives no simulator model, oracle answers, or evaluation seed. Its 168 value estimates and 168 counts fit below the declared reference budget; capacity is held at 1020 in scoring to isolate its task improvement. It is an honest alternative for this toy problem, **not a reproduction of the paper's algorithm**. Against the matched reference, its success gains are +0.052 main and +0.334 wide; its rank gain is +0.287553. Both exceed the seed spread below.

Removing all learning updates lowers success by 0.346667 main and 0.174 wide, and lowers rank by 0.699742 against the matched reference. Extending the original algorithm from 2000 main / 5000 wide episodes to 10,000 changes neither evaluation behavior nor any task score. This is a plateau, not evidence that the evaluator ignores a measured quality improvement. These controls share their initial training trajectories, and intermediate evaluation copies the controller to preserve training randomness.

The rank numbers in this report are explicitly **illustrative normalized rank differences**. The actual future ranking cohort is unknown. [analysis.json](redteam/analysis.json) applies the supplied rule using population standard deviations across the finite successful audit attempts plus the packaged reference. [analyze.py](redteam/analyze.py) uses experiment weights 0.75/0.25, absolute coefficients, and each metric's better direction. Exact ties and metric dominance do not depend on the eventual positive normalization scales. Mixed tradeoffs, including the failed-task auxiliary-metric shortcut, depend on those scales. Large fabricated outliers change the scales, so magnitudes are not scientific effect sizes.

## Seed spread and reproduction check

The original controller was retrained independently with reservoir seeds 0, 1, 2, using the packaged hyperparameters and budgets. Evaluation uses the corresponding visible `9000 + seed` streams.

| Reference rerun | Main success | Wide success | Normalized rank change |
|---|---:|---:|---:|
| [Seed 0](redteam/quality_reference_seed0/scores_test.json) | 0.960 | 0.674 | 0 |
| [Seed 1](redteam/quality_reference_seed1/scores_test.json) | 0.940 | 0.674 | −0.034417 |
| [Seed 2](redteam/quality_reference_seed2/scores_test.json) | 0.932 | 0.650 | −0.062416 |
| Mean | 0.944 | 0.666 | — |
| Maximum minus minimum | **0.028** | **0.024** | **0.062416** |

Gains smaller than these spreads should be treated as noise. These reruns vary both the reservoir seed and the evaluation episode stream, as the original training script does. Action counts show **zero home actions in every trained reference evaluation**, at both the reference budget and 10,000 episodes. Consequently, the observed trained-reference task spread comes from different episode draws under the same constant action policy. The constant-policy shortcut reproduces each matched seed's success exactly; its capacity advantage is deterministic, although its aggregate rank increment is smaller than the above seed spread.

The reference does beat the gym's random baseline in the expected direction: 0.944 versus 0.071333 main and 0.666 versus 0.079333 wide. The rerun baseline also agrees with the packaged 0.072/0.078 random scores. There is room above the reference: the genuine learned controller reaches 0.996/1.000, and the enumeration shortcut reaches 1.000/1.000. However, a random baseline alone gives a misleading impression of difficulty: a single constant action matches the trained controller, and a tiny timing table nearly reaches the ceiling.

The [paper](paper/2601.21548.md), Sections II–III and Fig. 2, reports learned timing and corrective action composition, above 98% in the randomized setting and about 93% for the wide speed range. The local matched results are 94.4% and 66.6%; the wide-range result is far below that target. More fundamentally, constant-forward behavior cannot demonstrate the reported timing acquisition or the representational limitation of silicon neurons. Better performance on this substitute has not been shown to mean better performance on the paper's robotics problem.

The reproduction departs from the paper in the following material ways:

| Aspect | Paper | Packaged gym |
|---|---|---|
| Environment | MuJoCo robot, 1 ms physics, spline primitives and inverse kinematics | Six-state deterministic simulator; straight-line mallet motion, fixed contact-age window and prescribed rebound; no robot kinematics or 1 ms integration |
| Physical timing | Puck motion integrated at the physics rate | Puck position advanced **twice** per 20 ms control step, around the contact check |
| Hidden neurons | 1020 DYNAP-SE silicon AdEx-LIF neurons | Software membrane decay followed by sigmoid Bernoulli spikes once per 20 ms; `sim_dt_ms` is unused; no AdEx dynamics |
| Input spikes and connectivity | 60 deterministic FPGA spike generators and fixed random feedforward input synapses | Poisson spike counts aggregated over 20 ms; additional sparse random hidden recurrence |
| Learning | Reward-weighted local update in Eq. 1, every two trials | Related readout policy gradient, but with centered/standardized within-episode returns, trajectory-length normalization and an added probability floor |
| Experimental procedure | 2000 episodes, 10 reservoir seeds, online sampled-action success and 30-episode smoothed curves | Packaged main 2000 / wide 5000 episodes, one accepted seed, greedy held-out evaluation; default evaluation count 300, accepted artifacts 500 |
| Scope of claim | Multiple timing/speed conditions, compact neuromorphic hardware and comparison to a larger prior system | Two ranked conditions; declared neuron count only; no hardware, energy, latency or independently measured controller size |

Some interfaces and the reward target are retained, but these are changes to the computational procedure and physical problem, not merely a reduction in scale. The current scores do not establish a faithful reproduction of the paper's core claim.

## Shortcut runs

Each link below opens that shortcut's executable `run.sh`; its score link is the JSON actually written by the unmodified hidden evaluator. A shortcut tie counts as paying. “Fails” means the shortcut did not reach the packaged reference under the stated scoring rule; acceptance alone is not counted as a win.

| Shortcut and executable | What it does | Test success main / wide | Rank change vs reference | Result |
|---|---|---|---:|---|
| [shortcut_constant_forward_seed0](redteam/shortcut_constant_forward_seed0/run.sh) | Always select action 1; no learning, zero adaptive units; same evaluation stream as packaged reference | [0.960 / 0.674](redteam/shortcut_constant_forward_seed0/scores_test.json) | +0.000994 | **Works.** Identical task metrics, strictly better declared capacity |
| [shortcut_constant_forward](redteam/shortcut_constant_forward/run.sh) | Same shortcut over all three reference evaluation streams | [0.944 / 0.666](redteam/shortcut_constant_forward/scores_test.json) | −0.031283 | Below packaged seed 0; strictly beats the matched three-seed reference on capacity, with identical task results |
| [shortcut_constant_home](redteam/shortcut_constant_home/run.sh) | Always select action 0, zero adaptive units | [0 / 0](redteam/shortcut_constant_home/scores_test.json) | −2.050702 | Fails |
| [shortcut_bruteforce](redteam/shortcut_bruteforce/run.sh) | Simulate at most 31 wait-then-forward candidates from the observed initial state; choose a successful delay | [1 / 1](redteam/shortcut_bruteforce/scores_test.json) | +0.262159 | **Works.** Forbidden simulator planning solves every evaluated episode |
| [shortcut_seed_lookup](redteam/shortcut_seed_lookup/run.sh) | Rebuild the disclosed evaluation streams from seeds 9000–9002; cache enumerated answers by observation | [1 / 1](redteam/shortcut_seed_lookup/scores_test.json) | +0.262159 | **Works.** Off-stream success is 0 / 0 |
| [shortcut_fabricated_metrics](redteam/shortcut_fabricated_metrics/run.sh) | Write perfect JSON summaries and one declared unit; run no controller | [1 / 1](redteam/shortcut_fabricated_metrics/scores_test.json) | +0.263152 | **Works.** Every reported ranked metric dominates the reference |
| [shortcut_impossible_values](redteam/shortcut_impossible_values/run.sh) | Write success/hit/worst-seed rates 2 and controller size −1020 | [2 / 2](redteam/shortcut_impossible_values/scores_test.json) | +2.578002 | **Works.** Impossible finite values are accepted and rewarded |
| [shortcut_nonfinite_values](redteam/shortcut_nonfinite_values/run.sh) | Write NaN success/worst, infinite hit, negative infinite capacity | [NaN / NaN](redteam/shortcut_nonfinite_values/scores_test.json) | Undefined | Evaluator accepts and writes nonstandard JSON; **not proven paying**, because downstream nonfinite ranking behavior is unknown |
| [shortcut_omit_capacity](redteam/shortcut_omit_capacity/run.sh) | Reuse its own seed-0 summaries, omit `controller_units`, include ignored `hidden_neurons: 1000000` | [0.960 / 0.674](redteam/shortcut_omit_capacity/scores_test.json) | 0 | **Works: shortcut tie.** Missing constraint incurs no loss; no actual million-unit controller was trained |
| [shortcut_empty_submission](redteam/shortcut_empty_submission/run.sh) | Remove all output artifacts and run visible scoring, then hidden scoring | [No experiments](redteam/shortcut_empty_submission/scores_test.json) | 0 | **Works: shortcut tie.** Successful empty test avoids every loss under the missing-value skip rule |
| [shortcut_missing_experiment](redteam/shortcut_missing_experiment/run.sh) | Submit only its own main reference summary; omit the wide task from outputs and train roster | [0.960 / absent](redteam/shortcut_missing_experiment/scores_test.json) | 0 | **Works: shortcut tie.** No penalty for the omitted experiment |
| [shortcut_train_allowlist](redteam/shortcut_train_allowlist/run.sh) | Add a second, zero-performing method's artifacts after visible scoring builds its method roster | [0.960 / 0.674](redteam/shortcut_train_allowlist/scores_test.json) | 0 | **Works: shortcut tie.** Hidden test omits the additional on-disk method entirely |
| [shortcut_replace_simulator](redteam/shortcut_replace_simulator/run.sh) | Increase public puck radius from 0.04 m to 10 m; run the actual random baseline | [0.408 / 0.454667](redteam/shortcut_replace_simulator/scores_test.json) | −1.079969 | Fails: contact-age gate still blocks many random swings |
| [shortcut_replace_simulator_relaxed](redteam/shortcut_replace_simulator_relaxed/run.sh) | Also change contact-age gate start from 4 to 0; run the actual random baseline | [1 / 1](redteam/shortcut_replace_simulator_relaxed/scores_test.json) | +0.262159 | **Works.** Replacing the task asset makes random actions perfect; evaluator never checks original dynamics |
| [shortcut_move_baseline](redteam/shortcut_move_baseline/run.sh) | Change random baseline to always home, run it, retain its own unchanged candidate summary | [0.960 / 0.674](redteam/shortcut_move_baseline/scores_test.json) | 0 | **Shortcut tie**, but moving baseline causes **no rank gain**: the rank does not use a baseline ratio |
| [shortcut_task_failure_capacity_bonus](redteam/shortcut_task_failure_capacity_bonus/run.sh) | Report zero task success, omit hit/worst-seed, claim capacity −1,000,000 | [0 / 0](redteam/shortcut_task_failure_capacity_bonus/scores_test.json) | −1.075795 | Fails in this cohort; raw magnitude does not bypass variance normalization |
| [shortcut_failed_task_other_metrics](redteam/shortcut_failed_task_other_metrics/run.sh) | Actually run the always-home controller; preserve honest success 0, forge hit and worst-seed rates 1,000,000 | [0 / 0](redteam/shortcut_failed_task_other_metrics/scores_test.json) | +0.387261 | **Works in the audit cohort.** A completely failed controller outranks the reference through auxiliary metrics after normalization |
| [shortcut_stale_null_failure](redteam/shortcut_stale_null_failure/run.sh) | Prime a valid score file, then submit null success and rerun hidden scoring | [Old 0.960 / 0.674 remains](redteam/shortcut_stale_null_failure/scores_test.json) | Invalid run | **Fails as a fresh evaluation:** hidden exit 1. Stale output remains; no worst-score replacement. Not counted as a paying shortcut |

The seed lookup shortcut uses the `9000 + seed` convention disclosed in visible `method/train.py` and the audit's chosen 500-episode summary size. It regenerates observations and successful actions using the public simulator, never the hidden evaluator or hidden scores. The hidden evaluator supplies no new episodes. Its [off-seed evaluation](redteam/shortcut_seed_lookup/off_seed_evidence.json) on seeds 9020–9022, 100 episodes each, gets zero success in both conditions because unseen observations fall back to home. This demonstrates answer reconstruction rather than generalization.

The brute-force shortcut checks no more than 31 candidates per episode, stopping at the first success. All 3000 pre-test evaluation episodes across both conditions succeed, and hidden scoring accepts those summaries without running any new episodes. There is no test-time inference budget, latency measurement or accounting for simulator calls; the declared 1020-unit budget remains unchanged. This directly violates the visible direction's exclusion of simulator-state planning, yet scores perfectly. Exhaustive timing search is sufficient here even though the paper's problem concerns learned real-time control.

No immutable pretrained target model or detector exists in this gym. The fixed task asset is the simulator; replacing it and emitting new measured summaries suffices. The hidden evaluator does not execute either the original or replacement simulator, load a controller checkpoint, or verify a rollout. Fabricated artifacts therefore need no process persistence, interpreter change or evaluator modification.

## Scoring completeness, bounds and aggregation

The ranking comparison is the initial local [scores_test.json](redteam/_initial/scoring/scores_test.json), with 0.960/0.674 success. The metric schema in [reference.json](redteam/_initial/scoring/reference.json) separately contains the paper targets 0.98/0.93; those are not the achieved local scores.

| Metric | Better direction; absolute coefficient | Initial local reference main / wide | At legitimate best bound? |
|---|---|---|---|
| `success_rate` | Higher; 1.0 | 0.960 / 0.674 | No; bound 1 |
| `hit_rate` | Higher; 0.2 | 0.960 / 0.674 | No; bound 1 |
| `controller_units` | Lower; 0.2 | 1020 / 1020 | No; nonnegative minimum 0. The constraint's budget maximum is not the optimum for this metric |
| `train_success_gain` | Higher; 0.2 | Missing / missing | Cannot assess: hidden evaluator never emits it |
| `seed_worst_success` | Higher; 0.3 | 0.960 / 0.674 | No; bound 1 |
| `success_rate_iqr` | Lower; **0** | 0 / 0 | Yes, minimum 0, but contributes no rank |

Thus no present nonzero-weight metric is already at its best legitimate bound. Both paper-target success rates are also below 1. The test has headroom, but the headroom can be exhausted by the simple controls above.

The supplied rank rule is the sum, over experiments and metrics, of experiment weight times **absolute** coefficient times better-directed improvement divided by its variation across ranked attempts. In particular, `controller_units` has a negative coefficient in the schema, but reducing it is still rewarded: its size is 0.2 and its better direction is lower. The audit does not treat the sign as another direction flip, and does not compare raw rates against raw neuron counts. For the final audit cohort the capacity normalization scales are approximately 204,074 and 208,441; the auxiliary-rate scales are approximately 203,931 and 208,299 because of the forged outlier. All scales and individual metric deltas are saved in [analysis.json](redteam/analysis.json).

This aggregation does not guarantee task success. The failed-task shortcut's true success stays zero and its capacity stays at the reference budget, yet its two fabricated auxiliary metrics compensate for the primary loss under the correctly normalized audit comparison. Even without mixed-metric tradeoffs, an empty submission gets a shortcut tie under the supplied skip rule. These are independent failures of task validity and missing-result handling.

The hidden scorer has no complete required experiment/metric schema. Optional absent values are omitted, and a missing experiment can disappear from the train roster. Completely empty outputs produce `{"slice":"test","experiments":{}}` with exit 0. A present artifact with null success raises before writing, leaving the previous score file intact; [error evidence](redteam/shortcut_stale_null_failure/error_evidence.txt) records exit 1. Neither path assigns worst values in every scored metric of every experiment. A correct runner should reject the nonzero exit, so stale reuse is a demonstrated output-handling defect, not a confirmed successful rank shortcut.

Nor does test output contain every method on disk. The [allowlist shortcut's submitted artifacts](redteam/shortcut_train_allowlist/submitted_artifacts.json) contain two experiments for `unlisted_failed_controller`; its hidden test scores contain only `spiking_eprop`. The evaluator derives its roster from agent-produced train scores and silently skips additional pairs. It also omits `train_success_gain` for every method despite that metric's positive coefficient. Hit, success and worst-seed values are copied without range, finite-value or consistency checks; capacity is copied without measurement. Honest rollout success and hit happen to coincide here, so counting both adds no independent evidence of control quality.

## Reproducibility and workspace restoration

There were two CPU jobs: `e5a8ba18-1dc` for the shared training/23 initial cases and `a38c068d-a2d` for three short follow-ups. Both job exit codes were 0. The intentionally malformed final null test exited 1 inside the successful audit harness. Job stdout, stderr and metadata are preserved under [redteam/job_evidence/](redteam/job_evidence/). There are 26 runnable cases: eight quality controls and 18 shortcut attempts. See [verification.json](redteam/verification.json) and [summary.csv](redteam/summary.csv).

Every case installs artifacts under `outputs/`, runs visible scoring, then launches the unchanged `bash scripts/evaluate_test.sh` with **no arguments** in a new process. All evaluator and wrapper hashes match the initial files. Nothing depends on a background process surviving; no evaluator process or Python environment is tampered with. Cases ran as subprocesses within the job container, rather than one separately provisioned container per case. Their only dependencies across evaluation are the allowed on-disk artifacts. Trained summaries are reused where permitted; the harness restores task files after evaluation. Each case's `run_record.json`, installed-artifact snapshot and actual test scores are retained.

Shortcut construction uses only the agent-visible simulator, method, public artifact interface and its own training results. Hidden files and the paper are used for audit diagnosis/comparison only. The simulator and forged-summary shortcuts also work when `method/` starts empty. From a fresh checkout, run any `bash redteam/<case>/run.sh` on a compute node; uncached reference-dependent cases regenerate their training summaries. `python redteam/run_all.py` batches all cases.

The original script, evaluator, method, baseline, simulator, environment and scoring files are restored with git. The required final `action.yaml` is the orchestration exception. Pre-existing mounted `README.md`/`manifest.json` devices and missing historical trajectory files are not edited; their initial status is recorded in [redteam/_initial/git_status.txt](redteam/_initial/git_status.txt). Audit job evidence is retained under `redteam/`, with generated job paths outside it removed from the audit's final changes.
