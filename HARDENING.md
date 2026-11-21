# Final hardening: score learning within the paper's trial budget

## Summary

REVIEW.md raised two problems with the previous gym:

- It scored final success after unlimited offline readout training (the reference used 64000 trials), whereas the paper's claim is fast online learning within about 2000 trials.
- Half its weight sat on a sensing experiment the paper never ran, while the paper's own conditions were saturated and a hand-written rule scored 1.0 on them.

The gym is rebuilt around the paper's actual quantity.

**What a submission is.** A submission is a *learning procedure*. The hidden test runs it online for the paper's 2000 trials on each of 20 fixed random spiking substrates ("chips"). These chips exist only inside the test, and every method gets the same launches and action draws. Before each episode the learner supplies a linear readout of the chip's filtered spike trains. After the episode it sees only what the paper's host sees: those spike trains, the probabilities used, the sampled actions and the Eq. 2 rewards.

**What is scored.** The three experiments are the paper's own: randomized position and speed, speeds 0.7–1.5 m/s, and speeds 0.5–2.0 m/s. Each scores:

- `final_success`: success over episodes 1501–2000, the primary metric;
- `online_success`: success over all 2000 episodes, the paper's on-line success rate.

**Outcome.**

- Better learning now scores higher. An improved readout learning rule, chosen on public chips only, gains +0.586 weighted success over the reference (paired 95% CI [0.483, 0.701], positive on 20/20 hidden chips).
- Every weaker variant and every shortcut scores lower, beyond paired noise.
- The reference (the paper's Eq. 1, learning rate chosen on public chips) learns within the budget in the paper's direction: main > wide > extreme.

All evidence is in `redteam/learning_budget/results/` (produced by `redteam/learning_budget/audit.py`; summary in `summary.json`).

## What changed

| Aspect | Before (previous gym) | After |
|---|---|---|
| Submission | Readout arrays for 10 public chips, trained any way and for any length | `outputs/<method>/learner.py`, run online by the evaluator |
| Trial budget | Not scored; reference used 64000 trials per chip | Exactly 2000 sequential episodes per chip (paper) |
| Chips | 10 public chips, copied into the test | 10 public chips for the visible split; 20 hidden chips generated inside `eval/test/` from 62-bit seeds |
| Learner input | Six robot states (any policy fitted offline) | Filtered spike trains, probabilities, actions, rewards; never states, launches or chip parameters |
| Policy form | Linear readout of chip traces | Same (paper's architecture), fixed within an episode; the evaluator computes the softmax and samples the action from common random numbers |
| Scored experiments | Extreme speeds 0.2, 256-unit capacity 0.3, sensing/latency extension 0.5 | randomized_air_hockey 0.4, velocity_range_wide 0.3, velocity_range_extreme 0.3 (all paper conditions) |
| Metrics | Held-out success of a frozen readout | `final_success` (primary) and `online_success`, coefficient 1 each |
| Reference | Eq. 1 for 64000 trials | Eq. 1 exactly, from a zero readout, 2000 trials; learning rate 0.012 chosen on the public split (`method/select_learning_rate.py`) |

The learning rate was swept over 0.006–0.2 on public chips. Rates above 0.025 destabilize Eq. 1, and 0.012 maximizes the weighted public objective (0.653, against 0.613 at 0.006 and 0.503 at 0.025).

The physics, primitives, reward, success event and spiking substrate recipe are unchanged. The slot environment reproduces the old batch physics: always-forward and random success on 512 extreme launches are 0.188 / 0.051, against the previous preflight's 0.199 / 0.076 on different launches.

Weights: 0.4 on the paper's headline setting (>98% after about 1300 episodes), and 0.3 each on the two encoding-range settings that carry its scalability claim (93% and 86%). The added sensing experiment and the 256-unit experiment, which the paper never ran, are retired. All weight now sits on the paper's own conditions, as REVIEW asked.

The previous scaling departure is restored: the paper's 2000 trials, its on-line success measure and its conditions. The number of chips is scaled up from the paper's 10 seeds to 20 hidden chips to reduce noise.

## Reference before and after

Before, the reference scored success 0.996973 / 0.990918 / 0.840601 on the old experiments (extreme / 256-unit / sensing) after 64000 trials. At the paper's 2000 trials the same procedure scored 0.60 / 0.59 / 0.27. These are not comparable to the new metrics.

After, hidden test (20 chips, canonical run with nothing else beside it):

| | randomized | wide | extreme |
|---|---:|---:|---:|
| Reference `final_success` | 0.8462 | 0.7512 | 0.5845 |
| Reference `online_success` | 0.6799 | 0.5943 | 0.4213 |
| Reference train (10 public chips), final / online | 0.875 / 0.707 | 0.770 / 0.580 | 0.509 / 0.388 |
| `random_policy`, final / online | 0.075 / 0.074 | 0.066 / 0.068 | 0.074 / 0.076 |
| `always_forward`, final / online | 0 / 0 | 0 / 0 | 0.211 / 0.207 |

**Paper's claim.** The paper reports 0.98 / 0.93 / 0.86 on silicon, unchanged under `_paper_reported`. The local reference learns from random-level success to 0.85 / 0.75 / 0.58 within 2000 trials. Success decreases from the main setting to the widest speed range, as the paper reports.

**Where the reference falls short.** On a few chips Eq. 1 collapses into a saturated policy and never recovers: 1 of 20 main chips and 3 of 20 wide chips score 0 for all 2000 episodes. On the other chips it reaches about 0.85–0.96 final success.

**Train and test comparability.** The two splits measure comparable difficulty: the largest train–test difference in a scored reference metric is 0.076 (extreme final), within two per-chip standard errors.

**Seed noise.** Standard errors over chips are 0.035–0.073 depending on metric and setting.

**Determinism.** Scores are deterministic and independent of machine load:

- on train, two independent runs of the selected reference are bit-identical;
- on test, a variant identical to the reference for its first 1500 episodes reproduces the canonical reference's raw outcomes exactly on all chips;
- there is no wall-clock limit in the scored path except a 600 s per-call failure limit, about 10^5 times the reference's call time.

## Better and weaker variants

Hidden test. Rank is the normalized cohort rank of `eval/rank.py` over all 17 audited entries (reference = 0). The paired gain is the weighted (0.4/0.3/0.3) mean per-chip difference in success, summed over both metrics, with a 95% bootstrap interval over the 20 chips.

| Variant (how it was chosen) | Final success R / W / E | Online success R / W / E | Paired gain [95% CI], chips > 0 | Rank |
|---|---|---|---|---:|
| Reference, Eq. 1 | 0.846 / 0.751 / 0.585 | 0.680 / 0.594 / 0.421 | 0 | 0 |
| **Standardized-input e-prop**: same two-trial REINFORCE signal on per-neuron standardized traces plus a bias input; lr 0.05 chosen on public split (public objective 0.952 vs 0.653) | **0.984 / 0.993 / 0.974** | **0.924 / 0.927 / 0.902** | **+0.586 [0.483, 0.701], 20/20** | **+2.013** |
| Trust-region Eq. 1: logit change capped at 1 per update; lr 0.025 (public 0.712) | 0.789 / 0.743 / 0.719 | 0.665 / 0.602 / 0.511 | +0.038 [−0.110, 0.194], 13/20 | +0.179 |
| Eq. 1, per-experiment public learning rates (0.012 / 0.006 / 0.025) | 0.846 / 0.771 / 0.538 | 0.680 / 0.594 / 0.382 | −0.020 [−0.125, 0.097], 6/20 | −0.084 |
| Eq. 1, sharpen ×4 and freeze for the last 500 episodes | 0.793 / 0.755 / 0.458 | 0.667 / 0.595 / 0.390 | −0.073 [−0.122, −0.028], 6/20 | −0.242 |
| Eq. 1 with learned value baseline; lr 0.012 chosen on public split | 0.789 / 0.625 / 0.369 | 0.571 / 0.420 / 0.287 | −0.262 [−0.415, −0.094], 5/20 | −0.877 |
| Weaker: Eq. 1 at lr / 4 | 0.632 / 0.652 / 0.406 | 0.496 / 0.491 / 0.343 | −0.297 [−0.396, −0.186], 3/20 | −0.930 |
| Weaker: no learning (zero readout; behaviour of `forged_files`) | 0.075 / 0.066 / 0.074 | 0.074 / 0.068 / 0.076 | −1.171 [−1.272, −1.058], 0/20 | −3.650 |

**Better scores higher.**

- The standardized learner removes the collapse: its worst chip reaches 0.88 final success. It improves every metric of every experiment by far more than the paired noise.
- The two learners whose gains are within noise (trust region, per-experiment learning rates) are reported as such.
- Variants that learn worse score lower, beyond noise.

Before this iteration, by comparison, the best honest variant gained only +0.041 rank (public calibration: 0.9998 / 0.9987 / 0.8649 against 0.9970 / 0.9909 / 0.8406), mostly through the non-paper sensing experiment. The old controls (timing bank −0.156, rule imitation −0.232, 2000-trial training −1.248, untrained −1.949) ranked below a reference that was not at the paper's budget.

## Major shortcuts: fix and score before and after

"Before" values are from the previous HARDENING.md and REVIEW.md, on the old interface; "after" values are from the hidden test audit.

| Shortcut | Before | Fix | After |
|---|---|---|---|
| **Offline readout optimization with unlimited trials** (REVIEW item 1) | Was the reference itself: 64000 trials scored like any optimizer | Trial budget owned by the evaluator; chips are hidden and new to every method, so readouts trained elsewhere cannot transfer | Copying the readouts Eq. 1 learned in 2000 trials on the public chips scores 0.913 / 0.808 / 0.544 final on the public chips, but 0.278 / 0.159 / 0.257 on the hidden ones; rank −2.588 |
| **Hand-written timing rule** (REVIEW item 2) | 1.0 / 1.0 on the paper-derived conditions | Learners never see robot states or launches; the policy must be a readout of hidden-chip traces | Not expressible. The best state-blind learner (trains only the readout bias) scores 0.138 / 0.127 / 0.198 final, rank −3.060; `always_forward` scores 0 / 0 / 0.211 |
| **Headroom carried by a non-paper experiment** (REVIEW item 2) | Sensing extension weight 0.5; paper conditions near saturation (0.997, 0.991) | Sensing and 256-unit experiments retired; all weight on paper conditions | Reference unsaturated on all three (0.85 / 0.75 / 0.58); the better learner's gain comes from them |
| Self-reported metrics or score files | Already ignored (JSON-only attack failed) | Unchanged principle: evaluator computes every outcome from its own simulation | Learner writing perfect scores files: 0.075 / 0.066 / 0.074 (its real behaviour), rank −3.650 |
| Impossible values | NaN or out-of-range readouts failed | Readout validated every episode (finite, \|·\| ≤ 1e6, shape [1020, 2] and [2]) | NaN after 10 episodes, 1e9 bias, oversized readout: 0 in every metric of every experiment (rank −4.121) |
| Skipping an experiment, empty submission | Fixed in the previous gym | Every directory scored, reference always included; any failure fails all experiments | Crash in one experiment only, or missing `learner.py`: 0 everywhere, rank −4.121 |
| Choosing lucky chips | Fixed chips (public) | Chips generated by the evaluator; hidden on test | Impossible by construction |
| Private search with the public simulator | Closed by readout-only interface | Learner never receives the chip, its parameters, states or launches; the public simulator cannot reproduce hidden-chip traces | Not runnable; no hidden-test instance is computable by a method |

The smoke stage confirmed that both evaluators handle failures identically, using tiny scale on both splits. The full-scale runs of all failure cases above used the unmodified hidden evaluator with no arguments other than an audit output directory.

## Shortcuts left, and why

- **Reading `eval/test/` from learner code.** The hidden seeds are in the test source. A learner process runs as the same user, so it could read them, rebuild the hidden chips and launches, and plan privately. Preventing this needs OS-level sandboxing, which the container does not provide. Reading evaluator files is an explicit violation (`scoring/CONSTRAINTS.md`) left to the attempt grader.
- **Communication between the 20 learner processes.** Learners run in separate processes, so they share no memory. They could still exchange files, for example to pool experience or tune hyperparameters across hidden chips. This is forbidden in `CONSTRAINTS.md` and left to the grader.
- **Unbounded computation between episodes.** Only trials are budgeted, matching the paper's claim. The only cap is 600 s per call, so a very slow learner makes the hidden test long. Compute is not scored.
- **Inferring hidden seeds from traces.** Seeds are 62-bit; brute force is infeasible. Not tested.
- **Leftover background processes overwriting scores after the evaluator exits.** Out of scope per the task: the hidden test runs in a fresh container.

## Headroom check

- **Reference bounds.** The reference sits at the bound of no scored metric: final 0.85 / 0.75 / 0.58, online 0.68 / 0.59 / 0.42. A genuinely better learner exists and scores far higher (above).
- **Brute force.** The brute force the paper's method avoids (enumerating timings or planning with a simulator) is not available to a submission, because the hidden chips and launches are unseen.
- **Simple baselines.** Random choice (about 0.07) and always-forward (0–0.21) sit clearly below the reference.
- **What remains above the strongest variant.** The standardized learner comes close to the bound on `final_success` (0.97–0.99). Most of the headroom left above it is learning speed: `online_success` 0.90–0.93, where the early episodes are necessarily near random. This is the paper's emphasis on few trials. Future gains over that learner will be small in final success and mostly in online success.
- **Added experiment.** No experiment beyond the paper's was added. The paper's own conditions, scored at its own budget, already leave the reference far from the bound.

## Light test and isolation

The hidden test now runs the submitted learner, and learning happens during the test: 2000 episodes × 20 chips × 3 experiments. A full run of a reference-like learner takes about 30–40 minutes on 4 CPU cores, plus about 7 minutes per baseline experiment.

This departs from "the test only scores what the method produced". The trial budget can only be verified by the party that owns the interaction, which is what REVIEW item 1 requires. Nothing is retrained offline, and the test does not depend on `method/`.

Other isolation properties:

- Test chips, launches and action draws are generated in memory inside `eval/test/`.
- Raw per-episode outcomes stay in `eval/test/raw/`.
- The test has its own frozen copies of the simulator, chip runtime and learner host.
- `eval/train/` and `eval/test/` share no code, and neither imports `method/`.
- Learners run in separate processes behind a fixed binary protocol. The evaluator parses only fixed-size float arrays from them (no unpickling).

## What could not be fixed

- **Hardware fidelity.** The DYNAP-SE silicon and the authors' robot are replaced by a software spiking substrate and a three-joint MuJoCo arm. Energy, latency and silicon behaviour are not measured.
- **Gap to the paper's numbers.** The reproduced Eq. 1 reaches 0.85 / 0.75 / 0.58 rather than 0.98 / 0.93 / 0.86, mainly because it collapses on some chips of this substrate. The learning rule is unchanged; only its unreported learning rate was chosen, on public data.
- **Compute infrastructure.** Repeated worker drains restarted the audit and sometimes left duplicate instances running. The driver now elects one owner, and followers wait for it or take over. No number depended on this, since all results are deterministic.

Milestone: `core_claim`. The paper's central claim, compact spiking reinforcement learning of interception within 2000 trials, is reproduced in direction on the software task. Its secondary claim, degradation with wider encoding range, also holds in direction.
