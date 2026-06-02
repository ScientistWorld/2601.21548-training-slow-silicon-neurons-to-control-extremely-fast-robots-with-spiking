#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p outputs checkpoints
COND="${CONDITION:-random_position_speed}"
OUT="${OUTPUT_PATH:-outputs/spiking_eprop_${COND}.json}"
python method/train.py \
  --condition "${COND}" \
  --episodes "${EPISODES:-2000}" \
  --eval-episodes "${EVAL_EPISODES:-400}" \
  --seeds "${SEEDS:-3}" \
  --hidden-neurons "${HIDDEN_NEURONS:-1020}" \
  --learning-rate "${LEARNING_RATE:-0.004}" \
  --entropy-floor "${ENTROPY_FLOOR:-0.08}" \
  --output "${OUT}"

bash scripts/evaluate.sh
