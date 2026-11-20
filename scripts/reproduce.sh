#!/bin/bash
# Reproduce the reference end to end: public learning-rate selection, install
# the Eq. 1 learner, visible scoring, held-out scoring.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
bash scripts/download.sh
bash scripts/method.sh --reselect
bash scripts/evaluate.sh
bash scripts/evaluate_test.sh
