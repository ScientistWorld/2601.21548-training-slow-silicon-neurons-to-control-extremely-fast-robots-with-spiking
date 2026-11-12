#!/bin/bash
# Run the paper's proposed method: fixed random spiking reservoir with online
# e-prop readout learning (Eq. 1), 2000 episodes, 10 reservoir seeds, for each
# scored condition.  Writes outputs/spiking_eprop/<experiment>/seed_<k>.npz and
# then scores them on the visible slice.
#
# Usage:
#   bash scripts/method.sh            # all experiments, seeds 0-9
#   bash scripts/method.sh --experiments randomized_air_hockey --seeds 0

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p outputs
python method/train.py --output-dir outputs/spiking_eprop --episodes "${EPISODES:-2000}" "$@"

bash scripts/evaluate.sh
