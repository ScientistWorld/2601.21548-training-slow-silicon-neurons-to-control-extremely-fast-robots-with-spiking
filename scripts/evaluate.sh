#!/bin/bash
# Evaluation script - the standard way to evaluate work in this environment.
#
# Executes every submitted controller found under outputs/<method>/ in closed
# loop on public validation episodes and writes scoring/scores.json.  Uses only
# eval/ and the public task assets in data/; it never imports method/.
#
# Submission contract (per method and experiment):
#   outputs/<method>/<experiment>/seed_<k>.npz   k = 0..4
# Each file is a controller configuration in the format documented in
# data/neuromorphic_chip.py.  Experiments: randomized_air_hockey,
# velocity_range_wide, velocity_range_extreme.
#
# Usage:
#   bash scripts/evaluate.sh

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

python eval/evaluate.py "$@"
