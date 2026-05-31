"""EvoPool: multi-label downstream training (BCE loss on RoBERTa-large).

Trains a multi-label classifier (HF ``problem_type="multi_label_classification"``)
on EvoPool annotator pseudo-labels. Supports WS pass, CFT (warm-up on noisy then
fine-tune on clean), and golden-only baselines.

Reads noisy WS labels from ``<run_root>/eval/train_labeled.jsonl`` (key
``aggregated_labels.majority_vote: List[int]``). Gold labels come from raw
``train.jsonl`` under ``true_labels: List[int]``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.downstream.metrics import compute_ml_metrics, hf_compute_metrics_ml
from src.utils.io import read_jsonl


def _set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def to_multihot(labels: List[int], K: int) -> np.ndarray:
    v = np.zeros(K, dtype=np.float32)
    for c in labels or []:
        try:
            ic = int(c)
            if 0 <= ic < K:
                v[ic] = 1.0
        except Exception:
            pass
    return v


class MLDataset(Dataset):
    def __init__(self, texts, labels_multihot, tokenizer, max_length):
        self.texts = texts
        self.y = labels_multihot
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.y[i], dtype=torch.float),
        }


def train_and_eval(
    train_texts: List[str], train_labels: np.ndarray,
    val_texts: List[str], val_labels: np.ndarray,
    test_texts: List[str], test_labels: np.ndarray,
    model_name: str, num_labels: int,
    epochs: int, batch_size: int, lr: float, max_length: int, seed: int,
    init_from: Optional[str] = None,
    grad_accum: int = 1,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        Trainer, TrainingArguments, set_seed as hf_set_seed,
    )
    hf_set_seed(seed)
    _set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        init_from or model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
    tr_ds = MLDataset(train_texts, train_labels, tok, max_length)
    va_ds = MLDataset(val_texts, val_labels, tok, max_length)
    te_ds = MLDataset(test_texts, test_labels, tok, max_length)

    work_dir = Path(os.environ.get("WORK_DIR", ".")) / "_trainer_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(work_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=seed, fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tr_ds,
                      eval_dataset=va_ds, compute_metrics=hf_compute_metrics_ml)
    trainer.train()

    def _probs(ds):
        out = trainer.predict(ds)
        return 1.0 / (1.0 + np.exp(-out.predictions))

    val_probs = _probs(va_ds)
    test_probs = _probs(te_ds)
    val_m = compute_ml_metrics(val_probs, val_labels)
    test_m = compute_ml_metrics(test_probs, test_labels)
    return trainer, val_m, test_m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", type=Path, required=True,
                   help="Path with eval/{train,val,test}_labeled.jsonl (noisy WS labels)")
    p.add_argument("--train_path", type=Path, required=True)
    p.add_argument("--val_path", type=Path, required=True)
    p.add_argument("--test_path", type=Path, required=True)
    p.add_argument("--model_name", default="FacebookAI/roberta-large")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--num_labels", type=int, required=True)
    p.add_argument("--mode", choices=["ws", "cft", "both", "golden_only"], default="both")
    p.add_argument("--clean_sizes", type=str, default="500,1000")
    p.add_argument("--ws_epochs", type=int, default=10)
    p.add_argument("--cft_epochs", type=int, default=5)
    p.add_argument("--ws_lr", type=float, default=2e-5)
    p.add_argument("--cft_lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    K = args.num_labels
    _set_seed(args.seed)

    train_gold = read_jsonl(args.train_path)
    val_gold = read_jsonl(args.val_path)
    test_gold = read_jsonl(args.test_path)
    train_texts_gold = [r["text"] for r in train_gold]
    train_y_gold = np.array([to_multihot(r.get("true_labels") or [], K) for r in train_gold])
    val_texts = [r["text"] for r in val_gold]
    val_y = np.array([to_multihot(r.get("true_labels") or [], K) for r in val_gold])
    test_texts = [r["text"] for r in test_gold]
    test_y = np.array([to_multihot(r.get("true_labels") or [], K) for r in test_gold])

    ws_path = args.run_root / "eval" / "train_labeled.jsonl"
    if not ws_path.exists():
        ws_path = args.run_root / "iter_00" / "eval" / "train_labeled.jsonl"
    train_y_noisy = None
    if args.mode in ("ws", "cft", "both"):
        assert ws_path.exists(), f"WS noisy labels missing: {ws_path}"
        ws_rows = read_jsonl(ws_path)
        id_to_noisy: Dict[str, List[int]] = {}
        for r in ws_rows:
            agg = r.get("aggregated_labels") or {}
            mv = agg.get("majority_vote")
            if isinstance(mv, list):
                id_to_noisy[str(r["id"])] = [int(x) for x in mv]
            elif isinstance(mv, int) and mv >= 0:
                id_to_noisy[str(r["id"])] = [mv]
            else:
                id_to_noisy[str(r["id"])] = []
        train_y_noisy = np.array([
            to_multihot(id_to_noisy.get(str(r["id"]), []), K) for r in train_gold
        ])
        print(f"[ws] noisy labels: avg cardinality "
              f"{train_y_noisy.sum(axis=1).mean():.2f} (gold avg "
              f"{train_y_gold.sum(axis=1).mean():.2f}), cov "
              f"{(train_y_noisy.sum(axis=1) > 0).mean():.3f}")

    results: Dict[str, Any] = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_classes": K,
    }

    if args.mode in ("ws", "cft", "both"):
        print(f"\n========== WS train (n={len(train_texts_gold)} noisy) ==========\n")
        t0 = time.time()
        trainer, vm, tm = train_and_eval(
            train_texts_gold, train_y_noisy, val_texts, val_y, test_texts, test_y,
            args.model_name, K, args.ws_epochs, args.batch_size, args.ws_lr,
            args.max_length, args.seed, grad_accum=args.grad_accum,
        )
        results["ws"] = {"val_metrics": vm, "test_metrics": tm,
                         "runtime_s": time.time() - t0}
        ws_ckpt = args.out_dir / "ws_model"
        try:
            trainer.save_model(str(ws_ckpt))
        except Exception as e:
            print(f"  [warn] saving ws_model failed: {e}")
        print(f"  WS  val macF1={vm['macro_f1']:.4f} test macF1={tm['macro_f1']:.4f}")

    if args.mode in ("golden_only", "both"):
        clean_sizes = [int(x) for x in args.clean_sizes.split(",")]
        results["golden_only"] = {}
        for N in clean_sizes:
            if N > len(train_gold):
                print(f"  [skip] golden N={N} > train size"); continue
            idx = np.random.RandomState(args.seed).choice(
                len(train_gold), size=N, replace=False)
            tx = [train_texts_gold[i] for i in idx]
            ty = train_y_gold[idx]
            print(f"\n========== Golden-only N={N} ==========\n")
            t0 = time.time()
            _, vm, tm = train_and_eval(
                tx, ty, val_texts, val_y, test_texts, test_y,
                args.model_name, K, args.cft_epochs, args.batch_size,
                args.cft_lr, args.max_length, args.seed, grad_accum=args.grad_accum,
            )
            results["golden_only"][f"clean_{N}"] = {
                "val_metrics": vm, "test_metrics": tm, "runtime_s": time.time() - t0,
            }
            print(f"  Golden N={N}  val macF1={vm['macro_f1']:.4f} "
                  f"test macF1={tm['macro_f1']:.4f}")

    if args.mode in ("cft", "both"):
        ws_ckpt = args.out_dir / "ws_model"
        if not ws_ckpt.exists():
            print("  [error] CFT requires WS first; ws_model missing"); sys.exit(2)
        clean_sizes = [int(x) for x in args.clean_sizes.split(",")]
        results["cft"] = {}
        for N in clean_sizes:
            if N > len(train_gold):
                print(f"  [skip] cft N={N} > train size"); continue
            idx = np.random.RandomState(args.seed).choice(
                len(train_gold), size=N, replace=False)
            tx = [train_texts_gold[i] for i in idx]
            ty = train_y_gold[idx]
            print(f"\n========== CFT clean N={N} (init from ws_model) ==========\n")
            t0 = time.time()
            _, vm, tm = train_and_eval(
                tx, ty, val_texts, val_y, test_texts, test_y,
                args.model_name, K, args.cft_epochs, args.batch_size,
                args.cft_lr, args.max_length, args.seed,
                init_from=str(ws_ckpt), grad_accum=args.grad_accum,
            )
            results["cft"][f"clean_{N}"] = {
                "val_metrics": vm, "test_metrics": tm, "runtime_s": time.time() - t0,
            }
            print(f"  CFT  N={N}  val macF1={vm['macro_f1']:.4f} "
                  f"test macF1={tm['macro_f1']:.4f}")

    out_path = args.out_dir / "ml_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n-> Wrote {out_path}")

    print("\n========== HEADLINE ==========")
    for cell_name in ("ws", "golden_only", "cft"):
        cell = results.get(cell_name)
        if not cell:
            continue
        if cell_name == "ws":
            tm = cell["test_metrics"]
            print(f"  WS              test macF1={tm['macro_f1']:.4f} "
                  f"microF1={tm['micro_f1']:.4f} weightedF1={tm['weighted_f1']:.4f}")
        else:
            for k, v in cell.items():
                tm = v["test_metrics"]
                print(f"  {cell_name:14s} {k}  test macF1={tm['macro_f1']:.4f} "
                      f"microF1={tm['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
