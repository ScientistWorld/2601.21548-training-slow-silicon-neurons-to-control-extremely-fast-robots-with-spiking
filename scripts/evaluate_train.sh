#!/bin/bash
# Visible train-slice evaluator.
#
# Mirrors scripts/evaluate.sh's artifact contract while writing
# scoring/scores_train.json. This script enumerates method artifacts from
# outputs/ by default and does not import or execute method code.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

python eval/train/evaluate.py "$@"
