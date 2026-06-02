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

python validate.py --reference-only
SEEDS=1 BASELINE_EPISODES=300 bash scripts/baseline.sh
SEEDS=1 EPISODES=1500 EVAL_EPISODES=300 HIDDEN_NEURONS=1020 bash scripts/method.sh
python validate.py --compare
