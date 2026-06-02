# Progress

## What Works

- Paper analysis is complete for the central claim.
- `data/air_hockey_env.py` defines the 6D air-hockey interception task, reward, success condition, and randomized benchmark condition.
- `method/` implements the paper's actual computational structure: Gaussian population-coded state spikes, fixed random LIF-style reservoir, filtered hidden traces, two-action softmax readout, and e-prop-style readout updates every two trials.
- `eval/` reads emitted method artifacts and writes `scoring/scores.json` without importing `method/`.
- Continuing-agent audit on 2026-06-02 verified that the implementation is not a surrogate, `scripts/evaluate.sh` works standalone, and `python validate.py --compare` passes all scoring and separation checks.
- Job `47cdad61-a3f` reached `core_claim`: the reproduced controller clearly beats the random sanity baseline on the randomized puck position/speed benchmark while preserving the 1020-neuron reservoir constraint.
- Job `b0836d70-c9d` extended the evaluator and score file to the paper's widest velocity-range setting, though this setting still needs improvement before it should count as `core_claim_plus`.
- Job `58fce6b9-d6d` tested a longer tuned wide-range run; it did not improve the final wide-range score, so the reusable scripts now keep the shorter run that reproduces the same score more cheaply.

## Results

- Job `7595de2e-7e4` produced `spiking_eprop` success rate 0.993 and hit rate 0.993 in the randomized condition. This result is not accepted as `core_claim` because the random sanity baseline also reached 1.0 success in the same simulator.
- Job `53701b5a-4a7` produced `spiking_eprop` success rate 0.953 and hit rate 0.953, but random action selection reached 0.973. This is also not accepted as `core_claim`.
- Job `c7cdbcc2-e98` produced random success 0.02, confirming the persistent-primitive task is non-trivial, but the method scored 0.0 after 2000 episodes.
- Job `47cdad61-a3f` produced random success 0.072 and `spiking_eprop` success 0.960 / hit rate 0.960 with 1020 hidden neurons. The paper reports >0.98 success in the corresponding randomized condition; this reproduction is close enough to support the central claim on the accessible task-level simulator.
- Job `b0836d70-c9d` reproduced the same randomized result and added `velocity_range_wide`: `spiking_eprop` success 0.674 / hit rate 0.674 with 1020 hidden neurons, compared with the paper's reported 0.930.
- Job `58fce6b9-d6d` kept `velocity_range_wide` at 0.674 after 5000 episodes with tuned hyperparameters. The random sanity artifact for the same wide condition scored 0.078.

## Remaining

- Higher milestones would require improving the wide velocity-range condition toward the paper's 0.930 result or adding more of the paper's reported ablations.
- Random sanity artifacts are available for each scored condition so future runs can check that the simulator has not become trivial.

## Issues

- The original DYNAP-SE hardware and MuJoCo robot stack are unavailable in this workspace, so the reproduction uses a software LIF/reservoir simulator and task-level air-hockey dynamics.
- Job `f65b0a96-f82` failed before baseline execution with `ModuleNotFoundError: No module named 'data'`. Cause: Python launched `baseline/random_policy.py` directly and did not include the workspace root on `sys.path`. Fixed by exporting `PYTHONPATH=/home/user` in the shell entry points.
- Job `7595de2e-7e4` exposed an invalid-evaluation issue: the task allowed a mallet to wait at the intercept point and still hit the puck, so random action selection was near-perfect. The simulator now requires active forward motion at contact.
- Job `53701b5a-4a7` showed the active-swing condition was still too weak because random switching often produced a valid forward motion near contact. The simulator now models persistent open-loop primitives and only counts contact during a narrow swing phase.
- Job `c7cdbcc2-e98` showed that the first persistent-primitive version made successes too sparse for the current online readout learner. The contact phase was widened and return centering was added to reduce policy-gradient variance without changing the local eligibility-trace computation.
- Job `47cdad61-a3f` resolved the non-triviality issue: random action selection stayed low while the paper method learned a high-success policy.
- Job `b0836d70-c9d` showed the wide speed distribution is harder for the current hyperparameters; the training-tail success was 0.72 and evaluation success was 0.674.
- Job `58fce6b9-d6d` showed that simply extending wide-range training to 5000 episodes with a higher learning rate and lower entropy floor does not improve evaluation success.

## Deviations from Paper

- Hardware substitution: the paper uses DYNAP-SE mixed-signal silicon neurons in the loop; this reproduction uses a software LIF-style random reservoir with the same input/readout/update structure because DYNAP-SE hardware is not available.
- Environment substitution: the paper uses the MuJoCo Air Hockey robot environment; this reproduction uses a tracked task-level simulator with the same 6D state, two primitive targets, contact threshold, reward, and success definition because the exact MuJoCo stack is not provided.
- Scale for first job: the submitted run uses fewer seeds than the paper's 10 independent reservoir samples to establish the core claim within the initial compute budget. The algorithm and training procedure are unchanged.
- Optimization stabilization: the reproduction centers discounted returns within each two-trial update batch. This is a policy-gradient variance-control addition around the paper's e-prop eligibility update, introduced because the software task-level simulator produced very sparse successes.
