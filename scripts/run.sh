#!/bin/bash
# Reproduce the final reference end to end; audit helpers are optional.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
bash scripts/reproduce.sh "$@"
python validate.py
