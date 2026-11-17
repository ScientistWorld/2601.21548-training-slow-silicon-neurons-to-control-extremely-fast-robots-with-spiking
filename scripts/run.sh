#!/bin/bash
# Combined audit retry: stability preflight, fresh calibration, all methods,
# isolated final scoring. Restore the end-to-end entry point after audit.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python redteam/final/study.py
