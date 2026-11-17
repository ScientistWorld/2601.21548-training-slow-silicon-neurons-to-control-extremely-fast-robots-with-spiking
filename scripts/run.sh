#!/bin/bash
# Resume the combined audit after node loss. Exact training checkpoints and
# source/input-keyed raw rollouts survive retries; scoring starts after training.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python redteam/final/study.py
