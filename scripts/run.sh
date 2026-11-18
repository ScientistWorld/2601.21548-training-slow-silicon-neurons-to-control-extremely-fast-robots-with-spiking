#!/bin/bash
# Finish the audit from the already-executed 64000-trial reference and cached
# raw rollouts. Calibration uses public inputs and exits before hidden scoring.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python redteam/final/check_paired.py
python redteam/final/finalize.py
