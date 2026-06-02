# Progress

## What Works

- Paper analysis is complete for the central claim.
- `data/air_hockey_env.py` defines the 6D air-hockey interception task, reward, success condition, and randomized benchmark condition.
- `method/` implements the paper's actual computational structure: Gaussian population-coded state spikes, fixed random LIF-style reservoir, filtered hidden traces, two-action softmax readout, and e-prop-style readout updates every two trials.
- `eval/` reads emitted method artifacts and writes `scoring/scores.json` without importing `method/`.

## Results

- Job `7595de2e-7e4` produced `spiking_eprop` success rate 0.993 and hit rate 0.993 in the randomized condition. This result is not accepted as `core_claim` because the random sanity baseline also reached 1.0 success in the same simulator.

## Remaining

- Rerun after the active-swing simulator fix and inspect both random sanity baseline and method artifacts.
- If the method success rate is not clearly above the random sanity baseline, tune learning rate, reservoir scale, or training length while preserving the paper's algorithm.
- Expand to velocity-range conditions after the core randomized setting is confirmed.

## Issues

- The original DYNAP-SE hardware and MuJoCo robot stack are unavailable in this workspace, so the reproduction uses a software LIF/reservoir simulator and task-level air-hockey dynamics.
- Job `f65b0a96-f82` failed before baseline execution with `ModuleNotFoundError: No module named 'data'`. Cause: Python launched `baseline/random_policy.py` directly and did not include the workspace root on `sys.path`. Fixed by exporting `PYTHONPATH=/home/user` in the shell entry points.
- Job `7595de2e-7e4` exposed an invalid-evaluation issue: the task allowed a mallet to wait at the intercept point and still hit the puck, so random action selection was near-perfect. The simulator now requires active forward motion at contact.

## Deviations from Paper

- Hardware substitution: the paper uses DYNAP-SE mixed-signal silicon neurons in the loop; this reproduction uses a software LIF-style random reservoir with the same input/readout/update structure because DYNAP-SE hardware is not available.
- Environment substitution: the paper uses the MuJoCo Air Hockey robot environment; this reproduction uses a tracked task-level simulator with the same 6D state, two primitive targets, contact threshold, reward, and success definition because the exact MuJoCo stack is not provided.
- Scale for first job: the submitted run uses fewer seeds than the paper's 10 independent reservoir samples to establish the core claim within the initial compute budget. The algorithm and training procedure are unchanged.
