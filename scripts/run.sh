#!/bin/bash
# Finish the combined audit from all 30 trained trajectories and saved raw
# rollouts. Unique temporary files make score writes safe across worker retries.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python redteam/final/study.py
