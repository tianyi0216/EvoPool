"""EvoPool: multi-label LoRA fine-tune for decoder LMs (Qwen3-1.7B, Llama-3.1-8B).

Mirrors ``train_decoder_lora.py`` for multi-label tasks (e.g. Kaggle PubMed,
Ohsumed-ML). Uses HuggingFace ``problem_type="multi_label_classification"`` ->
BCEWithLogitsLoss. Labels are multi-hot K-dim float tensors.

Modes (mirrors single-label):
  ws            : train on the full noisy aggregated train set
  cft           : ws warm-up, then fine-tune on N gold samples (clean_sizes)
  both          : ws + cft
  golden_only   : skip ws; train on N gold samples directly

Reads noisy annotator labels from ``<run_root>/eval/train_labeled.jsonl`` (key
``aggregated_labels.majority_vote: List[int]``). Gold labels live in raw
``train.jsonl`` under ``true_labels: List[int]``.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

from src.downstream.metrics import hf_compute_metrics_multilabel
from src.utils.io import read_jsonl


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_multihot(labels: List[int], K: int) -> List[float]:
    v = [0.0] * K
    for c in labels or []:
        try:
            ic = int(c)
            if 0 <= ic < K:
                v[ic] = 1.0
        except Exception:
            pass
    return v


class MLDataset(Dataset):
    def __init__(self, texts: List[str], labels_multihot: List[List[float]],
                 tokenizer, max_length: int):
        self.texts = texts
        self.y = labels_multihot
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_length)
        enc["labels"] = [float(x) for x in self.y[i]]
        return enc


def make_lora_model_ml(model_name: str, num_labels: int,
                       r: int = 16, alpha: int = 32, dropout: float = 0.05,
                       target_modules: Optional[List[str]] = None):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    print(f"\n[LoRA-ML] base = {model_name}, num_labels = {num_labels}")
    print(f"[LoRA-ML] r={r}, alpha={alpha}, dropout={dropout}, targets={target_modules}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        problem_type="multi_label_classification",
    )
    model.config.pad_token_id = tok.pad_token_id
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none",
        modules_to_save=["score"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, tok


def train_eval_ml(model, tok, train_texts, train_y, val_texts, val_y,
                  test_texts, test_y, *,
                  out_dir: Path, num_epochs: int, batch_size: int, grad_accum: int,
                  lr: float, max_length: int, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = MLDataset(train_texts, train_y, tok, max_length)
    val_ds = MLDataset(val_texts, val_y, tok, max_length)
    test_ds = MLDataset(test_texts, test_y, tok, max_length)
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.0,
        warmup_ratio=0.06,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="macro_f1",
        seed=seed, report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    collator = DataCollatorWithPadding(tok)
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, data_collator=collator,
        compute_metrics=hf_compute_metrics_multilabel,
    )
    print(f"\n[train-ml] {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    trainer.train()
    val_m = trainer.evaluate(eval_dataset=val_ds)
    test_m = trainer.evaluate(eval_dataset=test_ds)

    def strip(d):
        return {k[5:] if k.startswith("eval_") else k: v for k, v in d.items()}
    return strip(val_m), strip(test_m), trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", type=Path, required=True,
                   help="Path with eval/{train,val,test}_labeled.jsonl or iter_00/eval/.")
    p.add_argument("--train_path", type=Path, required=True)
    p.add_argument("--val_path", type=Path, required=True)
    p.add_argument("--test_path", type=Path, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--num_labels", type=int, required=True)
    p.add_argument("--mode", choices=["ws", "cft", "both", "golden_only"], default="both")
    p.add_argument("--clean_sizes", type=str, default="500,1000")
    p.add_argument("--ws_epochs", type=int, default=3)
    p.add_argument("--cft_epochs", type=int, default=5)
    p.add_argument("--ws_lr", type=float, default=1e-4)
    p.add_argument("--cft_lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    K = args.num_labels
    target_mods = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    # ── Load gold splits ──
    train_gold = read_jsonl(args.train_path)
    val_gold = read_jsonl(args.val_path)
    test_gold = read_jsonl(args.test_path)
    train_texts_gold = [r["text"] for r in train_gold]
    train_y_gold = [to_multihot(r.get("true_labels") or [], K) for r in train_gold]
    val_texts = [r["text"] for r in val_gold]
    val_y = [to_multihot(r.get("true_labels") or [], K) for r in val_gold]
    test_texts = [r["text"] for r in test_gold]
    test_y = [to_multihot(r.get("true_labels") or [], K) for r in test_gold]
    print(f"[load] train={len(train_gold)} val={len(val_gold)} test={len(test_gold)}")

    # ── Load noisy annotator labels ──
    ws_path = args.run_root / "eval" / "train_labeled.jsonl"
    if not ws_path.exists():
        ws_path = args.run_root / "iter_00" / "eval" / "train_labeled.jsonl"
    train_y_noisy: Optional[List[List[float]]] = None
    if args.mode in ("ws", "cft", "both"):
        assert ws_path.exists(), f"annotator labels missing: {ws_path}"
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
        train_y_noisy = [to_multihot(id_to_noisy.get(str(r["id"]), []), K)
                         for r in train_gold]
        avg_card_noisy = sum(sum(v) for v in train_y_noisy) / max(1, len(train_y_noisy))
        avg_card_gold = sum(sum(v) for v in train_y_gold) / max(1, len(train_y_gold))
        cov_noisy = sum(1 for v in train_y_noisy if sum(v) > 0) / max(1, len(train_y_noisy))
        print(f"[ws] noisy avg card {avg_card_noisy:.2f} (gold {avg_card_gold:.2f}), "
              f"cov {cov_noisy:.3f}")

    # ── noisy-label pass (full noisy train) ──
    if args.mode in ("ws", "cft", "both"):
        print("\n========== noisy-label pass (full noisy train) ==========")
        model, tok = make_lora_model_ml(args.model_name, K, args.lora_r,
                                        args.lora_alpha, args.lora_dropout,
                                        target_mods)
        ws_out = args.out_dir / "ws" / "ours_iter_00"
        val_m, test_m, ws_trainer = train_eval_ml(
            model, tok, train_texts_gold, train_y_noisy,
            val_texts, val_y, test_texts, test_y,
            out_dir=ws_out, num_epochs=args.ws_epochs, batch_size=args.batch_size,
            grad_accum=args.grad_accum, lr=args.ws_lr, max_length=args.max_length,
            seed=args.seed,
        )
        (ws_out / "results.json").write_text(json.dumps({
            "model_name": args.model_name, "num_train": len(train_texts_gold),
            "val_metrics": val_m, "test_metrics": test_m,
        }, indent=2))
        print(f"  noisy-label  val macF1={val_m['macro_f1']:.4f}  test macF1={test_m['macro_f1']:.4f}")
        ws_model_dir = args.out_dir / "ws_adapter"
        try:
            ws_trainer.save_model(str(ws_model_dir))
        except Exception as e:
            print(f"  [warn] saving noisy-trained adapter failed: {e}")
        del model, ws_trainer
        torch.cuda.empty_cache()

    # ── Golden-only cells (no WS) ──
    if args.mode in ("golden_only", "both"):
        print("\n========== Golden-only cells (clean N, no WS) ==========")
        clean_sizes = [int(x) for x in args.clean_sizes.split(",")]
        for N in clean_sizes:
            if N > len(train_gold):
                print(f"  [skip] golden N={N} > train size"); continue
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(train_gold), size=N, replace=False)
            tx = [train_texts_gold[i] for i in idx]
            ty = [train_y_gold[i] for i in idx]
            cell_out = args.out_dir / "cft" / f"golden_clean_{N}"
            model, tok = make_lora_model_ml(args.model_name, K, args.lora_r,
                                            args.lora_alpha, args.lora_dropout,
                                            target_mods)
            val_m, test_m, _ = train_eval_ml(
                model, tok, tx, ty, val_texts, val_y, test_texts, test_y,
                out_dir=cell_out, num_epochs=args.cft_epochs,
                batch_size=args.batch_size, grad_accum=args.grad_accum,
                lr=args.cft_lr, max_length=args.max_length, seed=args.seed,
            )
            (cell_out / "results.json").write_text(json.dumps({
                "model_name": args.model_name, "num_train": N,
                "val_metrics": val_m, "test_metrics": test_m,
            }, indent=2))
            print(f"  Golden N={N}  val macF1={val_m['macro_f1']:.4f}  "
                  f"test macF1={test_m['macro_f1']:.4f}")
            del model
            torch.cuda.empty_cache()

    # ── CFT cells (init from noisy-trained adapter, fine-tune on clean N) ──
    if args.mode in ("cft", "both"):
        ws_model_dir = args.out_dir / "ws_adapter"
        if not ws_model_dir.exists():
            print("  [warn] CFT requires noisy-label first -- ws_adapter missing; skipping CFT")
        else:
            clean_sizes = [int(x) for x in args.clean_sizes.split(",")]
            for N in clean_sizes:
                if N > len(train_gold):
                    print(f"  [skip] cft N={N} > train size"); continue
                rng = np.random.RandomState(args.seed)
                idx = rng.choice(len(train_gold), size=N, replace=False)
                tx = [train_texts_gold[i] for i in idx]
                ty = [train_y_gold[i] for i in idx]
                cell_out = args.out_dir / "cft" / f"cft_clean_{N}_iter_00"
                from peft import PeftModel
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    args.model_name, num_labels=K, torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    problem_type="multi_label_classification",
                )
                tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True,
                                                    trust_remote_code=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                base_model.config.pad_token_id = tok.pad_token_id
                model = PeftModel.from_pretrained(base_model, str(ws_model_dir),
                                                  is_trainable=True)
                model.print_trainable_parameters()
                val_m, test_m, _ = train_eval_ml(
                    model, tok, tx, ty, val_texts, val_y, test_texts, test_y,
                    out_dir=cell_out, num_epochs=args.cft_epochs,
                    batch_size=args.batch_size, grad_accum=args.grad_accum,
                    lr=args.cft_lr, max_length=args.max_length, seed=args.seed,
                )
                (cell_out / "results.json").write_text(json.dumps({
                    "model_name": args.model_name, "num_train": N,
                    "val_metrics": val_m, "test_metrics": test_m,
                }, indent=2))
                print(f"  CFT N={N}  val macF1={val_m['macro_f1']:.4f}  "
                      f"test macF1={test_m['macro_f1']:.4f}")
                del model, base_model
                torch.cuda.empty_cache()

    # ── Done marker for downstream pipelines ──
    cft_dir = args.out_dir / "cft"
    cft_dir.mkdir(parents=True, exist_ok=True)
    summary = {"model_name": args.model_name, "num_labels": K}
    (cft_dir / "curve_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> Wrote {cft_dir / 'curve_summary.json'} (done marker)")


if __name__ == "__main__":
    main()
