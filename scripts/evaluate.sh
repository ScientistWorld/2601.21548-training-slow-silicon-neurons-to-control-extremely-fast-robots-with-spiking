#!/bin/bash
# Visible scoring, with the canonical generic scores file for validation.
set -euo pipefail
cd /home/user
bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
