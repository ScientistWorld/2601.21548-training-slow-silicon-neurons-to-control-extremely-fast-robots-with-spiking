#!/bin/bash
# Run the next incremental reproduction setting from the paper.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring
mkdir -p outputs

CONDITION=speed_narrow BASELINE_EPISODES=500 SEEDS=1 bash scripts/baseline.sh
CONDITION=speed_narrow EPISODES=1000 EVAL_EPISODES=500 SEEDS=1 bash scripts/method.sh
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh

python - <<'PY'
import json
from pathlib import Path

scores = json.loads(Path("scoring/scores.json").read_text())
print(json.dumps(scores.get("experiments", {}).get("velocity_range_narrow", {}), indent=2))
PY
