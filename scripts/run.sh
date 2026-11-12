#!/bin/bash
# Reproduce the reference end to end (train the paper's controller for every
# experiment with 10 reservoir seeds, then score visible and held-out slices).
# The hardening study that produced HARDENING.md is
#   python redteam/hardening/run_hardening.py   (resumable by stage)

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

bash scripts/reproduce.sh
