#!/bin/bash
# Non-learning reference behaviours.
#
# The scored baselines (random_policy, always_forward) are computed inside the
# evaluators on the same episodes as every submitted method, so they appear in
# every scores file automatically.  This script runs the same behaviours on
# public episodes for inspection (logs/baselines.json).
#
# Usage:
#   bash scripts/baseline.sh

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

python baseline/random_policy.py "$@"
