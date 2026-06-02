#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"experiments": {"randomized_air_hockey": {"results": {"method": {"success_rate": 0.96}}}}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   bash scripts/evaluate.sh
#   bash scripts/evaluate.sh --artifacts outputs/spiking_eprop_random_position_speed.json
#
# Called from run.sh after training, or independently.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

python eval/evaluate.py "$@"
