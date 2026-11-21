# Experiments

Every experiment measures online learning of interception within the paper's
budget of 2000 sequential episodes per controller instance. The hidden test uses
20 fixed random instances and launch sequences that exist only inside the test;
every method sees the same instances, launches and action-sampling draws.
Two quantities are scored per experiment, averaged over instances:
`final_success` (success over the last 500 episodes: what has been learned by the
end of the budget) and `online_success` (success over all 2000 episodes: how
quickly it was learned). A method that fails anywhere scores 0 everywhere.
Both fixed comparisons (random primitive choice, always forward) run under the
same draws.

## randomized_air_hockey

The paper's main setting: launch position randomized within a 0.10 m window and
launch speed within 1.0–1.5 m/s. Tests whether reliable interception timing is
acquired from few trials when both where and how fast the puck arrives vary.

## velocity_range_wide

Launch speeds 0.7–1.5 m/s from a fixed position, encoded over a wider velocity
range by the same fixed number of input cells. Tests learning when the
substrate resolves the relevant variable more coarsely.

## velocity_range_extreme

Launch speeds 0.5–2.0 m/s, the widest range the paper reports, where it finds
the largest loss of success. Tests learning at the representational limit of the
fixed compact substrate.
