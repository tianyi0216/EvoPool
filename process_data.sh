#!/bin/bash
# process_data.sh — convert raw downloads into processed JSONL splits.
# Usage: bash process_data.sh [config.yaml]
set -e
CONFIG=${1:-config.yaml}
export PYTHONUNBUFFERED=1
python -m data.prepare --config "$CONFIG"
