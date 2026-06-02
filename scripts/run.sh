#!/bin/bash
# Verify the packaged ResearchGym scoring surface over existing artifacts.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring
rm -f scoring/scores.json scoring/scores_train.json scoring/scores_test.json

bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --compare
