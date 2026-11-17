Every experiment uses ten supplied chip instances. The experiment and seed
slot identify the instance; neither the synapses nor device parameters may
be replaced. Supply a full chip configuration with your own `w_out` and
`b_out`. Only those two arrays may change. Training is unrestricted: alternative
optimizers, longer learning, value estimation, and supervised readout fitting
are expressible. All train and test rollouts use the same frozen instances.
This isolates readout learning, the plastic component of the paper, and removes
chip selection as a source of apparent progress. No test episodes or answers
are used to calibrate these public assets.
