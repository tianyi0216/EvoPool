"""EvoPool: Aggregator entry-point. Reads project config.yaml, fits the chosen
aggregator on the latest pipeline iter, and dumps pseudo-labeled JSONL.

Config schema (in config.yaml):

    aggregator:
      method: evoagg          # evoagg | mv
      emb_model: sentence-transformers/all-MiniLM-L6-v2
      emb_cache: data/embeddings/chemprot_minilm_l6_v2.npz
      oof_folds: 5
      dump_dir: runs/chemprot_l0/evoagg_labels

    pipeline:
      out_root: runs/chemprot_l0   # used to locate iter_XX/eval

CLI:
    python -m src.aggregator.run_aggregator --config config.yaml [--eval_dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from src.tasks.configs import get_task_config
from src.utils.eval import read_jsonl

try:
    import yaml
except ImportError as e:
    raise SystemExit(
        "PyYAML is required to load config.yaml. Install with: pip install pyyaml"
    ) from e


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _resolve_eval_dir(cfg: Dict[str, Any], cli_eval_dir: Path) -> Path:
    if cli_eval_dir is not None:
        return cli_eval_dir
    pipeline = cfg.get("pipeline", {}) or {}
    out_root = pipeline.get("out_root")
    if out_root is None:
        raise SystemExit("config.pipeline.out_root not set and --eval_dir not provided.")
    out_root = Path(out_root)
    iter_dirs = sorted(out_root.glob("iter_*"))
    if not iter_dirs:
        # Some pipelines nest under pipeline/iter_XX
        nested = out_root / "pipeline"
        if nested.exists():
            iter_dirs = sorted(nested.glob("iter_*"))
    if not iter_dirs:
        raise SystemExit(f"No iter_* directory found under {out_root}.")
    return iter_dirs[-1] / "eval"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--eval_dir", type=Path, default=None,
                   help="Override: explicit iter_XX/eval dir. "
                        "If omitted, the latest iter under pipeline.out_root is used.")
    p.add_argument("--dump_dir", type=Path, default=None,
                   help="Override config.aggregator.dump_dir.")
    args = p.parse_args()

    cfg = _load_yaml(args.config)
    dataset_cfg = cfg.get("dataset", {}) or {}
    agg_cfg = cfg.get("aggregator", {}) or {}

    task_name = dataset_cfg.get("name")
    if not task_name:
        raise SystemExit("config.dataset.name not set.")
    task = get_task_config(task_name)
    multi_label = bool(getattr(task, "multi_label", False))

    method = (agg_cfg.get("method") or "evoagg").lower()
    eval_dir = _resolve_eval_dir(cfg, args.eval_dir)
    dump_dir = args.dump_dir or Path(agg_cfg.get("dump_dir") or (eval_dir.parent / f"{method}_labels"))

    print(f"[run-agg] task={task_name} method={method}")
    print(f"[run-agg] eval_dir={eval_dir}")
    print(f"[run-agg] dump_dir={dump_dir}")

    # Load splits
    rows_by_split: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        path = eval_dir / f"{split}_labeled.jsonl"
        if path.exists():
            rows_by_split[split] = read_jsonl(path)
            print(f"  loaded {split}: {len(rows_by_split[split])} rows")

    if method == "mv":
        from src.aggregator.majority_vote import dump_majority_vote_labels
        dump_majority_vote_labels(
            rows_by_split, dump_dir,
            num_classes=task.num_classes, multi_label=multi_label,
        )
        return

    if method != "evoagg":
        raise SystemExit(f"Unknown aggregator method: {method!r}. Expected 'evoagg' or 'mv'.")

    # EvoAgg path
    from src.aggregator.evoagg import (
        fit_predict_evoagg,
        fit_predict_evoagg_multilabel,
        dump_evoagg_labels,
    )
    emb_cache = agg_cfg.get("emb_cache")
    emb_path = Path(emb_cache) if emb_cache else None
    if emb_path is not None and not emb_path.exists():
        print(f"  [warn] emb_cache {emb_path} not found; EvoAgg will run votes-only")
        emb_path = None

    seed = int(cfg.get("pipeline", {}).get("seed", 42))
    n_folds = int(agg_cfg.get("oof_folds", 5))

    if multi_label:
        proba = fit_predict_evoagg_multilabel(
            train_rows=rows_by_split.get("train", []),
            val_rows=rows_by_split.get("val", []),
            test_rows=rows_by_split.get("test", []),
            num_classes=task.num_classes,
            emb_path=emb_path,
            seed=seed,
        )
    else:
        proba = fit_predict_evoagg(
            train_rows=rows_by_split.get("train", []),
            val_rows=rows_by_split.get("val", []),
            test_rows=rows_by_split.get("test", []),
            num_classes=task.num_classes,
            emb_path=emb_path,
            n_folds=n_folds,
            seed=seed,
        )

    dump_evoagg_labels(rows_by_split, proba, dump_dir, multi_label=multi_label)


if __name__ == "__main__":
    main()
