#!/bin/bash
# Finish the combined audit; resume raw scoring after worker preemptions.
# Restore the end-to-end reference entry point after the audit completes.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python redteam/final/study.py
