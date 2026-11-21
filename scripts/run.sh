#!/bin/bash
# Reproduce the reference end to end on a compute node (CPU only):
# public learning-rate selection (resumable), install the Eq. 1 learner,
# visible scoring, held-out scoring, then validation. The final hardening
# audit is redteam/learning_budget/audit.py (see HARDENING.md).
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
bash scripts/reproduce.sh
python validate.py
