#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p outputs scoring
rm -f outputs/spiking_eprop_*.json outputs/random_policy_*.json scoring/scores.json

python validate.py --reference-only
SEEDS=1 BASELINE_EPISODES=500 bash scripts/baseline.sh
SEEDS=1 EPISODES=2000 EVAL_EPISODES=500 HIDDEN_NEURONS=1020 bash scripts/method.sh
CONDITION=speed_wide SEEDS=1 BASELINE_EPISODES=500 bash scripts/baseline.sh
CONDITION=speed_wide SEEDS=1 EPISODES=5000 EVAL_EPISODES=500 HIDDEN_NEURONS=1020 LEARNING_RATE=0.006 ENTROPY_FLOOR=0.06 bash scripts/method.sh
python validate.py --compare
