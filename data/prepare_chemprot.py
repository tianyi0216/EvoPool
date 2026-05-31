#!/usr/bin/env python3
"""EvoPool: convert WRENCH ChemProt JSON to pipeline JSONL.

Input  : <raw_dir>/{train.json, valid.json, test.json}   (WRENCH ChemProt)
Output : <out_dir>/{train.jsonl, val.jsonl, test.jsonl}  +  label_map.json + dataset.json

Each output record carries:
    id, text, true_label (int), true_label_name, label, label_name,
    metadata.{source, original_split, original_idx, text_normalized,
              entity1, entity2, span1, span2, num_chars, num_words,
              wrench_weak_labels (optional)}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


LABEL_NAMES = {
    0: "Part-of",
    1: "Regulator",
    2: "Upregulator",
    3: "Downregulator",
    4: "Agonist",
    5: "Antagonist",
    6: "Modulator",
    7: "Cofactor",
    8: "Substrate/Product",
    9: "NOT",
}


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def convert_split(wrench_path: Path, split: str, out_dir: Path):
    with open(wrench_path) as f:
        data = json.load(f)

    out_path = out_dir / f"{split}.jsonl"
    records = []

    for orig_id, item in data.items():
        label = item["label"]
        d = item["data"]
        text = d["text"]
        entity1 = d.get("entity1", "")
        entity2 = d.get("entity2", "")
        span1 = d.get("span1", [])
        span2 = d.get("span2", [])

        record = {
            "id": f"chemprot_{split}_{orig_id}",
            "text": text,
            "true_label": label,
            "true_label_name": LABEL_NAMES[label],
            "label": label,
            "label_name": LABEL_NAMES[label],
            "metadata": {
                "source": "chemprot",
                "original_split": split,
                "original_idx": int(orig_id),
                "text_normalized": normalize_text(text),
                "entity1": entity1,
                "entity2": entity2,
                "span1": span1,
                "span2": span2,
                "num_chars": len(text),
                "num_words": len(text.split()),
            },
        }
        if "weak_labels" in item:
            record["metadata"]["wrench_weak_labels"] = item["weak_labels"]
        records.append(record)

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"  {split}: {len(records)} records -> {out_path}", flush=True)
    return records


def prepare_chemprot(raw_dir: str, out_dir: str, val_budget: int = 500) -> None:
    """Run the full ChemProt prep.

    Args:
        raw_dir   : directory containing WRENCH train.json / valid.json / test.json
        out_dir   : where to write {train,val,test}.jsonl + label_map.json + dataset.json
        val_budget: cap on validation split size (paper convention = 500)
    """
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_chemprot] raw_dir = {raw}", flush=True)
    print(f"[prepare_chemprot] out_dir = {out}", flush=True)

    split_files = [
        ("train", "train.json"),
        ("val", "valid.json"),
        ("test", "test.json"),
    ]

    dataset_info = {
        "name": "chemprot",
        "task_type": "single_label",
        "task_family": "classification",
        "num_classes": len(LABEL_NAMES),
        "label_names": LABEL_NAMES,
        "counts": {},
        "label_counts": {},
        "val_budget": val_budget,
    }

    for split, filename in split_files:
        records = convert_split(raw / filename, split, out)

        # Cap val split to budget (paper convention)
        if split == "val" and val_budget and len(records) > val_budget:
            records = records[:val_budget]
            with open(out / "val.jsonl", "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            print(f"  val: capped to {val_budget} records", flush=True)

        dataset_info["counts"][split] = len(records)
        dist = Counter(r["true_label"] for r in records)
        dataset_info["label_counts"][split] = {str(k): v for k, v in sorted(dist.items())}

    with open(out / "label_map.json", "w") as f:
        json.dump(LABEL_NAMES, f, indent=2)
    with open(out / "dataset.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"[prepare_chemprot] done. Label map + dataset.json saved to {out}/", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare ChemProt for EvoPool")
    ap.add_argument("--raw_dir", default="data/raw/chemprot",
                    help="Directory with WRENCH ChemProt {train,valid,test}.json")
    ap.add_argument("--out_dir", default="data/processed/chemprot",
                    help="Output directory for processed JSONL")
    ap.add_argument("--val_budget", type=int, default=500,
                    help="Cap val split to N rows (paper convention)")
    args = ap.parse_args()
    prepare_chemprot(args.raw_dir, args.out_dir, args.val_budget)


if __name__ == "__main__":
    main()
