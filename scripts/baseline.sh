#!/bin/bash
# Non-learning reference behaviours.
#
# The scored baselines (random_policy: forward with probability 0.5 every tick;
# always_forward) are run by the evaluators themselves on the same launches and
# action draws as every submitted learner, so they appear in every scores file.
# This script runs the same behaviours on public launches for inspection
# (logs/baselines.json).
set -e
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
python baseline/random_policy.py "$@"
