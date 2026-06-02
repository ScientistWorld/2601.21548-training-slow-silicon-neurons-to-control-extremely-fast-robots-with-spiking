#!/bin/bash
# Run the ResearchGym train/test scoring split over existing artifacts.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring
rm -f scoring/scores_train.json scoring/scores_test.json

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
