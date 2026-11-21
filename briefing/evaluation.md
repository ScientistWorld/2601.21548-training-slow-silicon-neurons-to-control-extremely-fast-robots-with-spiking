# Evaluation

**Submission.** A directory `outputs/<method>/` containing `learner.py` (plus any
files it reads from its own directory). `learner.py` defines
`make_learner(spec)`, returning an object with

- `readout() -> (w, b)`: the readout for the next episode, `w` of shape
  `[n_hidden, 2]` and `b` of shape `[2]` (action 0 = home, 1 = forward);
- `learn(episode) -> None`: called after every episode with a dict of numpy
  arrays: `trace [T, n_hidden]` (filtered hidden spike trains at each 20 ms step),
  `p_forward [T]` (forward probability used), `action [T]` (0/1 sampled),
  `reward [T]`.

`spec` holds `experiment`, `n_hidden` (1020), `n_actions` (2), `budget` (2000),
`chip` (instance index), `seed` and `workdir` (the submission directory).
At every step the evaluator computes `softmax(trace_t @ w + b)` and samples the
action from common random numbers. The readout is fixed within an episode.

**Protocol.** For each experiment the evaluator builds the controller
instances, starts one learner process per instance and runs 2000 sequential
episodes on each, asking for a readout before every episode. Instances learn
independently, in separate processes. The evaluator owns the robot, the chips,
the launches, the action draws and every outcome; all methods see identical
launches and draws on identical chips. The visible split uses 10 public chips
and launches; the hidden test uses 20 chips and launches that exist only inside
the test. Learning procedures, credit assignment, exploration, value or model
learning, and any computation between episodes are open.

**Metrics** (per experiment, averaged over instances, higher is better,
coefficient 1 each): `final_success`, the success fraction over episodes
1501–2000 (primary: success reached within the budget), and `online_success`,
the success fraction over all 2000 episodes (how fast success is acquired).

| Experiment | Launch x (m) | Speed (m/s) | Weight |
|---|---|---|---:|
| `randomized_air_hockey` | 0.95–1.05 | 1.0–1.5 | 0.4 |
| `velocity_range_wide` | 1.0 | 0.7–1.5 | 0.3 |
| `velocity_range_extreme` | 1.0 | 0.5–2.0 | 0.3 |

Rank: sum over experiments and metrics of weight x |coefficient| x
better-direction improvement over the reference, divided by the metric's
standard deviation across the ranked attempts (`eval/rank.py`).

**Failure.** A missing `learner.py`, an exception, a process exit, a readout of
the wrong shape, a nonfinite readout or one with any entry above 1e6 in magnitude,
or no answer within 600 s for a single call fails the whole method: it gets 0 in
every scored metric of every experiment. Files, numbers or flags a method writes
are never read.

**Running.** `bash scripts/evaluate_train.sh` scores every directory under
`outputs/` on the visible split (`scoring/scores_train.json`);
`bash scripts/evaluate_test.sh` (no arguments) scores them on the hidden test
(`scoring/scores_test.json`). For quick development,
`python eval/train/evaluate.py --methods <name> --budget 200 --chips 2` runs a
shortened visible check. Fixed baselines `random_policy` (forward with
probability 0.5 each step) and `always_forward` are run by the evaluator under
the same draws; these names are reserved. A full hidden-test run of a learner
like the reference takes about 30–40 minutes on 4 CPU cores.
