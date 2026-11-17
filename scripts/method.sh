#!/bin/bash
# Eq. 1 reference: 2000 trials, two trials per update, ten fixed chip instances.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
python scripts/generate_backbones.py
python method/train.py --output-dir outputs/spiking_eprop --episodes 2000 "$@"
