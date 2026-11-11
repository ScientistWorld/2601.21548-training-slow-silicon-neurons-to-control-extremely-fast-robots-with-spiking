#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user${PYTHONPATH:+:$PYTHONPATH}
# Install method artifacts, run visible scoring, then run the original hidden
# evaluator in a new process with no arguments.
python redteam/run_case.py quality_more_training
