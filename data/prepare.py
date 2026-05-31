#!/usr/bin/env python3
"""Top-level data prep dispatcher driven by config.yaml.

Usage: python -m data.prepare --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_config(path: str):
    try:
        from src.utils.config_loader import load_config  # type: ignore
        return load_config(path)
    except Exception:
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError(
                "PyYAML is required to read config.yaml. "
                "Install it via `pip install pyyaml`."
            ) from e
        with open(path) as f:
            return yaml.safe_load(f)


def _get(cfg, *keys, default=None):
    # Supports both nested dicts and SimpleNamespace.
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        if hasattr(cur, k):
            cur = getattr(cur, k)
        elif isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur if cur is not None else default


def dispatch(config_path: str) -> None:
    cfg = _load_config(config_path)

    name = _get(cfg, "dataset", "name")
    raw_dir = _get(cfg, "dataset", "raw_dir")
    processed_dir = _get(cfg, "dataset", "processed_dir")

    if not name:
        raise ValueError(f"config {config_path} is missing dataset.name")

    name = str(name).lower()
    processed_dir = processed_dir or f"data/processed/{name}"

    print(f"[prepare] dataset = {name}", flush=True)
    print(f"[prepare] raw_dir = {raw_dir}", flush=True)
    print(f"[prepare] processed_dir = {processed_dir}", flush=True)

    if name == "chemprot":
        from data.prepare_chemprot import prepare_chemprot
        if not raw_dir:
            raise ValueError("dataset.raw_dir is required for chemprot")
        val_budget = int(_get(cfg, "dataset", "val_budget", default=500))
        prepare_chemprot(raw_dir, processed_dir, val_budget=val_budget)

    elif name == "fever":
        from data.prepare_fever import prepare_fever
        max_train = int(_get(cfg, "dataset", "max_train", default=10000))
        max_val = int(_get(cfg, "dataset", "max_val", default=1000))
        max_test = int(_get(cfg, "dataset", "max_test", default=7600))
        seed = int(_get(cfg, "dataset", "seed", default=42))
        prepare_fever(
            out_dir=processed_dir,
            max_train=max_train,
            max_val=max_val,
            max_test=max_test,
            seed=seed,
        )

        if bool(_get(cfg, "dataset", "fever_enriched", default=False)):
            from data.enrich_fever import enrich_fever
            features = _get(cfg, "dataset", "fever_enrichment_features", default="A,B,C,D,E")
            enriched_dir = _get(
                cfg, "dataset", "fever_enriched_dir",
                default=f"{processed_dir}_enriched",
            )
            print(f"[prepare] enriching FEVER -> {enriched_dir}  (features={features})",
                  flush=True)
            enrich_fever(
                input_dir=processed_dir,
                output_dir=enriched_dir,
                features=features,
            )

    elif name == "pubmed":
        from data.prepare_pubmed import prepare_pubmed
        zip_path = _get(cfg, "dataset", "pubmed_kaggle_zip")
        if not zip_path:
            raise ValueError(
                "dataset.pubmed_kaggle_zip is required for pubmed "
                "(download the Kaggle PubMed multilabel zip first)"
            )
        val_budget = int(_get(cfg, "dataset", "val_budget", default=500))
        seed = int(_get(cfg, "dataset", "seed", default=42))
        prepare_pubmed(
            zip_path=zip_path,
            out_dir=processed_dir,
            seed=seed,
            val_budget=val_budget,
        )

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: chemprot | fever | pubmed."
        )

    print(f"[prepare] done. Processed files under {Path(processed_dir).resolve()}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="EvoPool data prep dispatcher")
    ap.add_argument("--config", default="config.yaml",
                    help="Path to EvoPool config.yaml")
    args = ap.parse_args()
    try:
        dispatch(args.config)
    except Exception as e:
        print(f"[prepare] ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
