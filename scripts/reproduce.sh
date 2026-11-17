#!/bin/bash
set -euo pipefail
cd /home/user
bash scripts/download.sh
bash scripts/method.sh "$@"
bash scripts/evaluate.sh
bash scripts/evaluate_test.sh
