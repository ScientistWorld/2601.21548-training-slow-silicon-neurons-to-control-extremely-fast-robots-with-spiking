#!/bin/bash
# Held-out test-slice evaluator (no arguments).
#
# Scores every method directory under outputs/ (plus the proposed methods of
# scoring/reference.json): each method's learner.py learns online for the
# paper's 2000-trial budget on hidden controller instances and launches, with
# frozen private copies of the task simulator and chip runtime; writes
# scoring/scores_test.json. Does not import method/.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

python eval/test/evaluate.py
