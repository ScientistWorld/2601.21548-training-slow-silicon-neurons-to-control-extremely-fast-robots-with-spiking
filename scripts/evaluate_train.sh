#!/bin/bash
# Visible train-slice evaluator.
#
# Scores every method directory under outputs/: each method's learner.py learns
# online for the paper's 2000-trial budget on the public controller instances
# and launches; writes scoring/scores_train.json. Does not import method/.
# Development options of eval/train/evaluate.py (--outputs, --methods, --budget,
# --chips) are not used here.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

python eval/train/evaluate.py
