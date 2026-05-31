#!/bin/bash
# run_evopool.sh — run the 4-agent annotator-evolution pipeline, then aggregate.
# Usage: bash run_evopool.sh [config.yaml]
set -e
CONFIG=${1:-config.yaml}
export PYTHONUNBUFFERED=1
echo "[1/2] Running EvoPool pipeline..."
python -m src.pipeline.run --config "$CONFIG"
echo "[2/2] Running aggregator..."
python -m src.aggregator.run_aggregator --config "$CONFIG"
