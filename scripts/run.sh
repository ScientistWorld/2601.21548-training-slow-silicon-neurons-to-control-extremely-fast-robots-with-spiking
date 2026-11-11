#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user${PYTHONPATH:+:$PYTHONPATH}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
python redteam/run_all.py
