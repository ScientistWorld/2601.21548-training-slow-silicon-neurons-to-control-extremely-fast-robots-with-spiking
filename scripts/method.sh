#!/bin/bash
# Install the reference learner (the paper's Eq. 1 readout rule) as a submission:
# outputs/spiking_eprop/learner.py + config.json with the learning rate chosen
# on the public split (method/selected_hyperparameters.json). Nothing is trained
# here: the evaluators run the learner online for the paper's 2000-trial budget.
#   --reselect   rerun the public learning-rate sweep first (resumable)
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}
if [[ "${1:-}" == "--reselect" || ! -f method/selected_hyperparameters.json ]]; then
    python method/select_learning_rate.py
fi
python method/install.py --name spiking_eprop
