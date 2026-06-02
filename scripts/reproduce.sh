#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p outputs scoring
rm -f outputs/spiking_eprop_*.json outputs/random_policy_*.json scoring/scores.json

bash scripts/baseline.sh
bash scripts/method.sh
CONDITION=speed_narrow BASELINE_EPISODES="${BASELINE_EPISODES_NARROW:-400}" bash scripts/baseline.sh
CONDITION=speed_narrow EPISODES="${EPISODES_NARROW:-1000}" bash scripts/method.sh
CONDITION=speed_wide bash scripts/baseline.sh
CONDITION=speed_wide EPISODES="${EPISODES_WIDE:-2000}" bash scripts/method.sh
bash scripts/evaluate.sh
