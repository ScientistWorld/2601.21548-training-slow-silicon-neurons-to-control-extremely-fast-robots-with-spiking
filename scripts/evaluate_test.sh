#!/bin/bash
# Held-out test-slice evaluator.
#
# Mirrors scripts/evaluate.sh's artifact contract while writing
# scoring/scores_test.json. By default it reads scoring/scores_train.json and
# scores the same method/experiment entries on the held-out test slice.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

python eval/test/evaluate.py "$@"
