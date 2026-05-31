#!/bin/bash
# eval.sh — eval annotator pool or downstream checkpoint.
#
# Usage:
#   bash eval.sh [config.yaml] <mode> [dir]
#
#   <mode> = annotator | ckpt
#   [dir]  = run directory   (annotator mode, optional; defaults to pipeline.out_root from config)
#          = checkpoint dir  (ckpt mode, REQUIRED; HuggingFace AutoModel checkpoint)
#
# Examples:
#   bash eval.sh                                              # config.yaml + annotator + default run_dir
#   bash eval.sh config.yaml annotator                         # explicit config + annotator
#   bash eval.sh config.yaml annotator runs/chemprot_l0        # + explicit run_dir
#   bash eval.sh config.yaml ckpt runs/chemprot_l0/downstream/ws/model
set -e

CONFIG=${1:-config.yaml}
MODE=${2:-annotator}
DIR=${3:-}

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [ "$MODE" = "annotator" ]; then
    EXTRA=""
    if [ -n "$DIR" ]; then EXTRA="--run_dir $DIR"; fi
    python -m src.utils.eval_runner --config "$CONFIG" --mode annotator $EXTRA
elif [ "$MODE" = "ckpt" ]; then
    if [ -z "$DIR" ]; then
        echo "ERROR: ckpt mode requires a checkpoint directory as the 3rd argument."
        echo "Usage: bash eval.sh [config.yaml] ckpt <ckpt_dir>"
        exit 1
    fi
    python -m src.utils.eval_runner --config "$CONFIG" --mode ckpt --ckpt_dir "$DIR"
else
    echo "ERROR: unknown mode '$MODE'. Use 'annotator' or 'ckpt'."
    echo "Usage: bash eval.sh [config.yaml] <annotator|ckpt> [dir]"
    exit 1
fi
