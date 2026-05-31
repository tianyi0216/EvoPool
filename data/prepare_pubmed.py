#!/usr/bin/env python3
"""Parse the Kaggle PubMed 14-class multi-label zip to pipeline JSONL.

Source: user-supplied Kaggle "PubMed Multi Label Text Classification" zip.
Auto-detects the largest CSV, the text column, and binary 0/1 label columns
(each capital-letter MeSH category).
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_label_map(path: Path, label_map: Dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in label_map.items()}, indent=2))


def prepare_pubmed(
    zip_path: str,
    out_dir: str,
    val_frac: float = 0.1,
    train_frac: float = 0.8,
    seed: int = 42,
    val_budget: int = 500,
) -> None:
    import pandas as pd

    zip_p = Path(zip_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not zip_p.exists():
        raise FileNotFoundError(f"Kaggle PubMed zip not found: {zip_p}")

    print(f"[prepare_pubmed] reading {zip_p}", flush=True)
    with zipfile.ZipFile(zip_p, "r") as zf:
        csv_names = [
            n for n in zf.namelist()
            if n.lower().endswith(".csv") and "__macosx" not in n.lower()
        ]
        if not csv_names:
            raise RuntimeError(f"No CSV inside zip: {zip_p}; contents: {zf.namelist()[:5]}")
        csv_names.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
        csv_name = csv_names[0]
        print(f"[prepare_pubmed] parsing CSV {csv_name} "
              f"({zf.getinfo(csv_name).file_size//1024} KB)", flush=True)
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    print(f"[prepare_pubmed] loaded df shape={df.shape}", flush=True)

    text_candidates = [
        "abstractText", "abstract", "abstract_text", "text",
        "Abstract", "Text",
    ]
    text_col = next((c for c in text_candidates if c in df.columns), None)
    if text_col is None:
        if "Title" in df.columns and "abstractText" in df.columns:
            df["__text__"] = df["Title"].fillna("") + " " + df["abstractText"].fillna("")
            text_col = "__text__"
        else:
            raise RuntimeError(f"Could not find text column; df columns: {list(df.columns)}")
    print(f"[prepare_pubmed] text column: {text_col}", flush=True)

    skip_cols = {
        text_col, "Title", "abstractText", "abstract", "id", "ID",
        "meshMajor", "meshMinor", "PMID", "pmid",
    }
    label_cols = [
        c for c in df.columns
        if c not in skip_cols
        and df[c].dropna().isin([0, 1, True, False, "0", "1"]).all()
    ]
    if not label_cols:
        raise RuntimeError(f"No binary 0/1 label columns found; df cols: {list(df.columns)}")
    print(f"[prepare_pubmed] {len(label_cols)} label columns: {label_cols}", flush=True)

    label_map = {i: name for i, name in enumerate(label_cols)}
    write_label_map(out / "label_map.json", label_map)

    all_rows = []
    for idx, row in df.iterrows():
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        labels = []
        for i, c in enumerate(label_cols):
            v = row[c]
            if pd.notna(v):
                truthy = bool(v) if isinstance(v, bool) else int(v) == 1
                if truthy:
                    labels.append(i)
        all_rows.append({
            "id": f"pubmed_{idx}",
            "text": text,
            "true_labels": labels,
            "true_label_names": [label_map[c] for c in labels],
            "metadata": {
                "source": f"kaggle:{zip_p.name}",
                "num_chars": len(text),
                "num_words": len(text.split()),
            },
        })

    rng = random.Random(seed)
    rng.shuffle(all_rows)
    n = len(all_rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    val_split = all_rows[n_train:n_train + n_val]
    if val_budget and len(val_split) > val_budget:
        val_split = val_split[:val_budget]
    splits_out = {
        "train": all_rows[:n_train],
        "val":   val_split,
        "test":  all_rows[n_train + n_val:],
    }

    dataset_info = {
        "name": "pubmed",
        "task_type": "multi_label",
        "task_family": "classification",
        "source_zip": str(zip_p),
        "label_names": label_map,
        "num_classes": len(label_cols),
        "seed": seed,
        "counts": {},
        "label_counts": {},
        "avg_cardinality": {},
        "val_budget": val_budget,
    }

    for s, rows in splits_out.items():
        write_jsonl(out / f"{s}.jsonl", rows)
        cnt = Counter(c for r in rows for c in r["true_labels"])
        avg_card = sum(len(r["true_labels"]) for r in rows) / max(1, len(rows))
        dataset_info["counts"][s] = len(rows)
        dataset_info["label_counts"][s] = {str(k): v for k, v in sorted(cnt.items())}
        dataset_info["avg_cardinality"][s] = round(avg_card, 3)
        print(f"  {s}: {len(rows)} rows, avg_card={avg_card:.2f}", flush=True)

    with open(out / "dataset.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"[prepare_pubmed] done. Files saved to {out}/", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Kaggle PubMed multilabel for EvoPool")
    ap.add_argument("--kaggle_pubmed_zip", required=True,
                    help="Path to user-downloaded Kaggle PubMed multilabel zip")
    ap.add_argument("--out_dir", default="data/processed/pubmed")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_budget", type=int, default=500)
    args = ap.parse_args()
    prepare_pubmed(
        zip_path=args.kaggle_pubmed_zip,
        out_dir=args.out_dir,
        val_frac=args.val_frac,
        train_frac=args.train_frac,
        seed=args.seed,
        val_budget=args.val_budget,
    )


if __name__ == "__main__":
    main()
