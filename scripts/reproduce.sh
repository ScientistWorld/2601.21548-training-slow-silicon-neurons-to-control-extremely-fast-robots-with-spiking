#!/bin/bash
# Reproduce the reference end-to-end: train the paper's controller for every
# scored condition (5 reservoir seeds, 2000 episodes each) and score it on the
# visible and held-out slices.  Baselines are computed by the evaluators.
#
# Single CPU: roughly 1-2 hours for training plus ~15 minutes of scoring.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

bash scripts/download.sh
mkdir -p outputs scoring
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
