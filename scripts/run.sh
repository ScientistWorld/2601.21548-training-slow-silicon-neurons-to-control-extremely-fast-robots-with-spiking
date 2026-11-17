#!/bin/bash
# One combined scientific audit; all training finishes before hidden scoring.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python redteam/final/study.py
