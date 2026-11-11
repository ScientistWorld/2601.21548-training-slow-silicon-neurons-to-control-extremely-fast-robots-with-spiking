#!/bin/bash
# Hardening job: reproduce the reference with the restored paper settings, run
# better/weaker variants and shortcut cases through the unmodified evaluators,
# and regenerate the reference-only scoring files.  Resumable by stage.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

python redteam/hardening/run_hardening.py
