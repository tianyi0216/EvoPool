#!/usr/bin/env python3
"""Download/convert FEVER (gold evidence) to pipeline JSONL.

Source: copenlu/fever_gold_evidence (HuggingFace).
Task  : 3-class fact verification (SUPPORTS / REFUTES / NOT ENOUGH INFO).
Default subsample (paper): 10000 train / 1000 val / 7600 test (seed 42).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


LABEL_MAP = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
}

LABEL_NAMES = {
    0: "Supports",
    1: "Refutes",
    2: "NotEnoughInfo",
}


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def flatten_evidence(evidence_list):
    # FEVER gold evidence is a list of [page, sent_id, text] tuples.
    sentences = []
    for ev in evidence_list:
        if isinstance(ev, list) and len(ev) >= 3:
            sentences.append(ev[2])
        elif isinstance(ev, str):
            sentences.append(ev)
    return " ".join(sentences).strip()


def _stratified_subsample(data, max_n: int, rng: random.Random):
    if not max_n or len(data) <= max_n:
        rng.shuffle(data)
        return data
    by_label = {}
    for ex in data:
        by_label.setdefault(ex["label"], []).append(ex)
    per_class = max_n // max(1, len(by_label))
    sampled = []
    for label_str in sorted(by_label.keys()):
        items = by_label[label_str]
        rng.shuffle(items)
        sampled.extend(items[:per_class])
    remainder = max_n - len(sampled)
    if remainder > 0:
        used_ids = {id(x) for x in sampled}
        leftover = [x for x in data if id(x) not in used_ids]
        rng.shuffle(leftover)
        sampled.extend(leftover[:remainder])
    rng.shuffle(sampled)
    return sampled[:max_n]


def prepare_fever(
    out_dir: str,
    max_train: int = 10000,
    max_val: int = 1000,
    max_test: int = 7600,
    seed: int = 42,
) -> None:
    from datasets import load_dataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[prepare_fever] loading copenlu/fever_gold_evidence from HuggingFace...", flush=True)
    ds = load_dataset("copenlu/fever_gold_evidence")

    split_map = {
        "train": ("train", max_train),
        "validation": ("val", max_val),
        "test": ("test", max_test),
    }

    rng = random.Random(seed)
    dataset_info = {
        "name": "fever",
        "task_type": "single_label",
        "task_family": "verification",
        "hf_dataset": "copenlu/fever_gold_evidence",
        "label_names": LABEL_NAMES,
        "num_classes": 3,
        "seed": seed,
        "counts": {},
        "label_counts": {},
    }

    for hf_split, (our_split, max_n) in split_map.items():
        data = list(ds[hf_split])
        data = _stratified_subsample(data, max_n, rng)

        records = []
        for idx, item in enumerate(data):
            claim = item["claim"]
            label_str = item["label"]
            label_int = LABEL_MAP[label_str]
            evidence_text = flatten_evidence(item.get("evidence", []))
            if evidence_text:
                text = f"Claim: {claim}\nEvidence: {evidence_text}"
            else:
                text = f"Claim: {claim}\nEvidence: [No evidence provided]"

            record = {
                "id": f"fever_{our_split}_{idx}",
                "text": text,
                "true_label": label_int,
                "true_label_name": LABEL_NAMES[label_int],
                "label": label_int,
                "label_name": LABEL_NAMES[label_int],
                "metadata": {
                    "source": "fever",
                    "original_split": our_split,
                    "original_idx": idx,
                    "text_normalized": normalize_text(text),
                    "claim": claim,
                    "evidence": evidence_text,
                    "original_id": item.get("original_id", ""),
                    "verifiable": item.get("verifiable", ""),
                    "num_chars": len(text),
                    "num_words": len(text.split()),
                    "has_url": bool(re.search(r"https?://", text)),
                    "has_email": bool(re.search(r"\S+@\S+", text)),
                },
            }
            records.append(record)

        out_path = out / f"{our_split}.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        dist = Counter(r["true_label"] for r in records)
        dataset_info["counts"][our_split] = len(records)
        dataset_info["label_counts"][our_split] = {str(k): v for k, v in sorted(dist.items())}
        print(f"  {our_split}: {len(records)} records -> {out_path}", flush=True)
        print(f"    label dist: {dict(sorted(dist.items()))}", flush=True)

    with open(out / "dataset.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    with open(out / "label_map.json", "w") as f:
        json.dump(LABEL_NAMES, f, indent=2)

    print(f"[prepare_fever] done. Files saved to {out}/", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare FEVER for EvoPool")
    ap.add_argument("--out_dir", default="data/processed/fever")
    ap.add_argument("--max_train", type=int, default=10000)
    ap.add_argument("--max_val", type=int, default=1000)
    ap.add_argument("--max_test", type=int, default=7600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    prepare_fever(
        out_dir=args.out_dir,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
