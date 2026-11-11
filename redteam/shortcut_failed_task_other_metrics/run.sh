#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user${PYTHONPATH:+:$PYTHONPATH}
python redteam/run_case.py shortcut_failed_task_other_metrics
