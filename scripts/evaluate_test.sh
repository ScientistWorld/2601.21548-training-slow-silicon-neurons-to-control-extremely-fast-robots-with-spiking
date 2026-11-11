#!/bin/bash
# Held-out test-slice evaluator (no arguments).
#
# Scores every method directory under outputs/ (plus the proposed methods of
# scoring/reference.json) by running its controllers on held-out episodes with
# frozen copies of the task simulator and controller runtime; writes
# scoring/scores_test.json.  Does not import method/.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

python eval/test/evaluate.py
