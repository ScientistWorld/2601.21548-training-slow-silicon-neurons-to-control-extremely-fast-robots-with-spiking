#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

mkdir -p outputs checkpoints
python method/train.py \
  --condition "${CONDITION:-random_position_speed}" \
  --episodes "${EPISODES:-2000}" \
  --eval-episodes "${EVAL_EPISODES:-400}" \
  --seeds "${SEEDS:-3}" \
  --hidden-neurons "${HIDDEN_NEURONS:-1020}" \
  --output outputs/spiking_eprop_random_position_speed.json

bash scripts/evaluate.sh
