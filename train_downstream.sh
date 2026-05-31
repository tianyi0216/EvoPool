#!/bin/bash
# train_downstream.sh — train a downstream classifier on the aggregated labels.
# Usage: bash train_downstream.sh [config.yaml]
set -e
CONFIG=${1:-config.yaml}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m src.downstream.train --config "$CONFIG"
