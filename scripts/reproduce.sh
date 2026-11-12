#!/bin/bash
# Reproduce the reference end-to-end: train the paper's controller for every
# scored experiment (10 reservoir seeds, 2000 episodes each) and score it on the
# visible and held-out slices.  Baselines are computed by the evaluators.
#
# Training: 40 runs of 2000 episodes (about 2-4 CPU-minutes each); scoring
# runs (method, experiment) pairs in parallel over the available CPUs.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

bash scripts/download.sh
mkdir -p outputs scoring
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
