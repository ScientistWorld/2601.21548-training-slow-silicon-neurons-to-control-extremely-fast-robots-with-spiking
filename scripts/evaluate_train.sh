#!/bin/bash
# Visible train-slice evaluator.
#
# Scores every method directory under outputs/ (plus the proposed methods of
# scoring/reference.json) by running its controllers on public validation
# episodes; writes scoring/scores_train.json.  Does not import method/.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

python eval/train/evaluate.py
