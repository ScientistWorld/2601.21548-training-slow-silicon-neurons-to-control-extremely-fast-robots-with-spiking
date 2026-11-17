# Experiments

Every experiment evaluates ten assigned controller instances on 4096 identical
paired launches per instance. It reports task success, weakest-instance
success, physical contact frequency and measured controller size. Only task
success contributes to rank. Invalid submissions receive worst values across
all experiments. All baselines use the same launches and observation noise.

## velocity_range_extreme

Tests successful forward interception across the broad 0.5–2.0 m/s speed range
where the paper reports losses. The 1020-unit limit retains the paper's scale.

## single_core_extreme

Tests the same range under a 256-unit limit. It asks how well learning uses a
restricted fixed controller capacity. Violating that capacity fails the run.

## sensor_robustness

Tests forward interception across wider launch positions and oblique incoming
headings, with 0.01 m position noise, 0.10 m/s velocity noise, and one control
tick observation latency. This extends the sensing and mechanical robustness
questions raised by the paper, where timing is uncertain and simple launch
schedules are less reliable. It carries half the experiment weight.

## Fixed comparisons

Every experiment includes a uniformly random action policy, an immediate
forward commitment, and the strongest global member of a fixed bank of short intercept-time rules.
The bank uses eight fixed lead times plus a distance-based travel-time estimate;
reporting its best global member favors this simple comparison conservatively. These measure
whether improved success reflects learning beyond basic control schedules.
They are computed by the evaluator, never read from submitted logs.
