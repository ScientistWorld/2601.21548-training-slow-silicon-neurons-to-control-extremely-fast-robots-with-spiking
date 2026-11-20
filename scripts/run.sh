#!/bin/bash
# Final hardening job: public learning-rate selection, isolated reference and
# baseline scoring, then the hidden-test audit of variants and shortcuts.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python redteam/learning_budget/audit.py
python validate.py
