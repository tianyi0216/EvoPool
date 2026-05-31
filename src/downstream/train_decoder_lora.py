"""EvoPool: single-label LoRA fine-tune for decoder LMs (Qwen3-1.7B, Llama-3.1-8B).

Mirrors the CLI of ``train_encoder.py`` so downstream launchers can dispatch on
``--model_name``. Production LoRA defaults: r=16, alpha=32, dropout=0.05,
target_modules=[q_proj, k_proj, v_proj, o_proj]; bf16; no quantization (an L40
48GB GPU is enough for both Qwen3-1.7B and Llama3-8B at batch=4-16).

Modes:
  ws            : single-pass training on the noisy aggregated labels
  cft           : ``ws`` followed by clean fine-tuning at each --clean_sizes N
  both          : ws + cft cells
  self_training : 3-stage canonical ST (N golden -> pseudo-label train ->
                  fresh LoRA on pseudo -> CFT on N golden)
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

from src.downstream.metrics import compute_metrics
from src.utils.io import read_jsonl


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── dataset ────────────────────────────────────────────────────────────


class TextDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer, label_key: str,
                 max_length: int = 128, label_extractor=None):
        self.rows = rows
        self.tok = tokenizer
        self.label_key = label_key
        self.max_length = max_length
        self.label_extractor = label_extractor or (lambda r: int(r[label_key]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r.get("text", ""), truncation=True, max_length=self.max_length)
        enc["labels"] = self.label_extractor(r)
        return enc


# ── LoRA model factory ─────────────────────────────────────────────────


def make_lora_model(model_name: str, num_labels: int,
                    r: int = 16, alpha: int = 32, dropout: float = 0.05,
                    target_modules: Optional[List[str]] = None):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    print(f"\n[LoRA] base = {model_name}")
    print(f"[LoRA] r={r}, alpha={alpha}, dropout={dropout}, targets={target_modules}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
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


# ── train+eval helper ──────────────────────────────────────────────────


def train_eval(model, tok, train_rows, val_rows, test_rows, *,
               out_dir: Path, num_labels: int,
               num_epochs: int, batch_size: int, grad_accum: int,
               lr: float, max_length: int, seed: int,
               label_key_train: str = "label",
               label_extractor_train=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = TextDataset(train_rows, tok, label_key_train, max_length,
                           label_extractor=label_extractor_train)
    val_ds = TextDataset(val_rows, tok, "true_label", max_length)
    test_ds = TextDataset(test_rows, tok, "true_label", max_length)

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
        metric_for_best_model="f1_macro",
        seed=seed, report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    collator = DataCollatorWithPadding(tok)
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, data_collator=collator,
        compute_metrics=compute_metrics,
    )
    print(f"\n[train] {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    trainer.train()
    val_m = trainer.evaluate(eval_dataset=val_ds)
    test_m = trainer.evaluate(eval_dataset=test_ds)

    def strip(d):
        return {k[5:] if k.startswith("eval_") else k: v for k, v in d.items()}

    return strip(val_m), strip(test_m)


def predict_hard_labels(model, tok, rows: List[dict],
                        batch_size: int = 16, max_length: int = 128) -> List[int]:
    """Predict argmax class labels for each row (used in self_training Stage 2)."""
    from torch.utils.data import DataLoader
    model.eval()
    device = next(model.parameters()).device
    ds = TextDataset(rows, tok, label_key="true_label", max_length=max_length,
                     label_extractor=lambda r: int(r.get("true_label", 0)))
    collator = DataCollatorWithPadding(tok)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator, num_workers=2)
    preds: List[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(**batch)
            preds.extend(torch.argmax(out.logits.float(), dim=-1).cpu().tolist())
    model.train()
    return preds


def stratified_clean_sample(val_rows: List[dict], n_clean: int, num_labels: int,
                            seed: int) -> List[dict]:
    by_cls: Dict[int, List[dict]] = {c: [] for c in range(num_labels)}
    for r in val_rows:
        c = int(r.get("true_label", -1))
        if c >= 0 and c in by_cls:
            by_cls[c].append(r)
    per_class = max(n_clean // num_labels, 1)
    rng = random.Random(seed)
    out = []
    for c in range(num_labels):
        pool = by_cls.get(c, [])
        sampled = rng.sample(pool, min(per_class, len(pool)))
        for r in sampled:
            new_r = dict(r)
            new_r["label"] = new_r["true_label"]
            out.append(new_r)
    rng.shuffle(out)
    return out[:n_clean]


# ── main ───────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", type=Path, required=True,
                   help="contains iter_00/eval/train_labeled.jsonl with noisy labels")
    p.add_argument("--train_path", type=Path, required=True,
                   help="data/processed/<task>/train.jsonl (raw, used for text)")
    p.add_argument("--val_path", type=Path, required=True)
    p.add_argument("--test_path", type=Path, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--num_labels", type=int, required=True)
    p.add_argument("--mode", choices=["ws", "cft", "both", "self_training"], default="both")
    p.add_argument("--n_clean_for_st", type=int, default=500,
                   help="Number of val golden labels for self_training mode.")
    p.add_argument("--clean_sizes", type=str, default="25,50,100,200,500,1000")
    p.add_argument("--ws_epochs", type=int, default=3)
    p.add_argument("--cft_epochs", type=int, default=5)
    p.add_argument("--ws_lr", type=float, default=1e-4)
    p.add_argument("--cft_lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Qwen3-1.7B can do 16; Llama3-8B should use 4 on L40")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj")
    args = p.parse_args()

    _set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    iter0_labeled = args.run_root / "iter_00" / "eval" / "train_labeled.jsonl"
    if not iter0_labeled.exists():
        iter0_labeled = args.run_root / "eval" / "train_labeled.jsonl"
    noisy_rows = read_jsonl(iter0_labeled)
    val_rows = read_jsonl(args.val_path)
    test_rows = read_jsonl(args.test_path)
    print(f"Loaded noisy_train={len(noisy_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    def noisy_label(r):
        agg = r.get("aggregated_labels") or {}
        lab = agg.get("majority_vote", agg.get("mv", -1))
        return int(lab) if lab is not None else -1

    covered = [r for r in noisy_rows if noisy_label(r) >= 0]
    print(f"Coverage on noisy train: {len(covered)}/{len(noisy_rows)} "
          f"({len(covered)/max(len(noisy_rows),1):.3f})")

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]

    # ── WS pass ──
    if args.mode in ("ws", "both"):
        print(f"\n{'='*64}\n  WS pass on {len(covered)} noisy-labeled examples\n{'='*64}")
        model, tok = make_lora_model(args.model_name, args.num_labels,
                                     r=args.lora_r, alpha=args.lora_alpha,
                                     dropout=args.lora_dropout,
                                     target_modules=target_modules)
        val_m, test_m = train_eval(
            model, tok, covered, val_rows, test_rows,
            out_dir=args.out_dir / "ws",
            num_labels=args.num_labels,
            num_epochs=args.ws_epochs, batch_size=args.batch_size,
            grad_accum=args.grad_accum, lr=args.ws_lr,
            max_length=args.max_length, seed=args.seed,
            label_key_train="aggregated_labels",
            label_extractor_train=noisy_label,
        )
        ws_results = {"val_metrics": val_m, "test_metrics": test_m,
                      "config": {"model_name": args.model_name,
                                 "phase": "ws", "n_train_covered": len(covered),
                                 "lr": args.ws_lr, "epochs": args.ws_epochs,
                                 "lora_r": args.lora_r}}
        (args.out_dir / "ws" / "results.json").write_text(json.dumps(ws_results, indent=2))
        print(f"\n[WS] val_macF1={val_m.get('f1_macro'):.4f} "
              f"test_macF1={test_m.get('f1_macro'):.4f}")
        del model, tok
        torch.cuda.empty_cache()

    # ── CFT sweep ──
    if args.mode in ("cft", "both"):
        clean_sizes = [int(x) for x in args.clean_sizes.split(",") if x.strip()]
        print(f"\n{'='*64}\n  CFT sweep -- clean_sizes = {clean_sizes}\n{'='*64}")
        cft_summary = {}
        for n_clean in clean_sizes:
            clean_rows = stratified_clean_sample(val_rows, n_clean, args.num_labels,
                                                 args.seed)
            cell_dir = args.out_dir / "cft" / f"clean_{n_clean}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--- CFT clean={n_clean} ---")
            model, tok = make_lora_model(args.model_name, args.num_labels,
                                         r=args.lora_r, alpha=args.lora_alpha,
                                         dropout=args.lora_dropout,
                                         target_modules=target_modules)
            print(f"  [phase A] WS on {len(covered)} noisy examples")
            train_eval(model, tok, covered, val_rows, test_rows,
                       out_dir=cell_dir / "phaseA_ws",
                       num_labels=args.num_labels,
                       num_epochs=args.ws_epochs, batch_size=args.batch_size,
                       grad_accum=args.grad_accum, lr=args.ws_lr,
                       max_length=args.max_length, seed=args.seed,
                       label_extractor_train=noisy_label)
            print(f"  [phase B] CFT on {len(clean_rows)} clean examples")
            val_m, test_m = train_eval(
                model, tok, clean_rows, val_rows, test_rows,
                out_dir=cell_dir, num_labels=args.num_labels,
                num_epochs=args.cft_epochs, batch_size=args.batch_size,
                grad_accum=args.grad_accum, lr=args.cft_lr,
                max_length=args.max_length, seed=args.seed,
                label_key_train="label",
            )
            cft_summary[n_clean] = {"val_metrics": val_m, "test_metrics": test_m}
            (cell_dir / "results.json").write_text(json.dumps(
                {"val_metrics": val_m, "test_metrics": test_m,
                 "config": {"model_name": args.model_name, "n_clean": n_clean,
                            "phase": "cft", "ws_lr": args.ws_lr, "cft_lr": args.cft_lr,
                            "lora_r": args.lora_r}}, indent=2))
            print(f"  [clean={n_clean}] val_macF1={val_m.get('f1_macro'):.4f} "
                  f"test_macF1={test_m.get('f1_macro'):.4f}")
            del model, tok
            torch.cuda.empty_cache()
        (args.out_dir / "cft" / "curve_summary.json").write_text(
            json.dumps(cft_summary, indent=2))
        print(f"\nCFT curve summary written -> {args.out_dir/'cft'/'curve_summary.json'}")

    # ── self_training (3-stage) ──
    if args.mode == "self_training":
        N = args.n_clean_for_st
        print(f"\n{'='*64}\n  SELF-TRAINING 3-stage (n_clean={N})\n{'='*64}")
        train_rows_raw = read_jsonl(args.train_path)
        print(f"  Loaded raw train: {len(train_rows_raw)} rows")
        clean_rows = stratified_clean_sample(val_rows, N, args.num_labels, args.seed)
        print(f"  Stratified {len(clean_rows)} clean rows from val")

        # Stage 1: train LoRA on N val golden labels -> model_1
        print(f"\n--- Stage 1: train on {len(clean_rows)} golden labels ---")
        stage1_dir = args.out_dir / "stage_1"
        model_1, tok_1 = make_lora_model(args.model_name, args.num_labels,
                                         r=args.lora_r, alpha=args.lora_alpha,
                                         dropout=args.lora_dropout,
                                         target_modules=target_modules)
        val_m1, test_m1 = train_eval(
            model_1, tok_1, clean_rows, val_rows, test_rows,
            out_dir=stage1_dir, num_labels=args.num_labels,
            num_epochs=args.ws_epochs, batch_size=args.batch_size,
            grad_accum=args.grad_accum, lr=args.ws_lr,
            max_length=args.max_length, seed=args.seed,
            label_key_train="label",
        )
        (stage1_dir / "results.json").write_text(json.dumps(
            {"val_metrics": val_m1, "test_metrics": test_m1,
             "config": {"model_name": args.model_name, "phase": "stage_1_golden",
                        "n_clean": N, "lora_r": args.lora_r}}, indent=2))
        print(f"  [stage_1] val_macF1={val_m1.get('f1_macro'):.4f} "
              f"test_macF1={test_m1.get('f1_macro'):.4f}")

        # Stage 2 (predict): model_1 pseudo-labels all train
        print(f"\n--- Stage 2 predict: pseudo-label {len(train_rows_raw)} train rows ---")
        pseudo_preds = predict_hard_labels(
            model_1, tok_1, train_rows_raw,
            batch_size=args.batch_size * 2, max_length=args.max_length)
        from collections import Counter as _C
        print(f"  Pseudo-label distribution: {dict(_C(pseudo_preds))}")
        pseudo_rows = []
        for r, p_ in zip(train_rows_raw, pseudo_preds):
            new_r = dict(r)
            new_r["aggregated_labels"] = {"majority_vote": int(p_),
                                          "_self_training_pseudo": True}
            pseudo_rows.append(new_r)
        pseudo_path = args.out_dir / "pseudo_train.jsonl"
        with pseudo_path.open("w") as f:
            for r in pseudo_rows:
                f.write(json.dumps(r) + "\n")
        print(f"  Wrote pseudo train: {pseudo_path}")
        del model_1, tok_1
        torch.cuda.empty_cache()

        # Stage 2 (train): fresh LoRA on pseudo train
        print(f"\n--- Stage 2 train: fresh LoRA on {len(pseudo_rows)} pseudo-labeled rows ---")
        stage2_dir = args.out_dir / "stage_2"
        model_2, tok_2 = make_lora_model(args.model_name, args.num_labels,
                                         r=args.lora_r, alpha=args.lora_alpha,
                                         dropout=args.lora_dropout,
                                         target_modules=target_modules)

        def pseudo_label(r):
            return int(r["aggregated_labels"]["majority_vote"])

        val_m2, test_m2 = train_eval(
            model_2, tok_2, pseudo_rows, val_rows, test_rows,
            out_dir=stage2_dir, num_labels=args.num_labels,
            num_epochs=args.ws_epochs, batch_size=args.batch_size,
            grad_accum=args.grad_accum, lr=args.ws_lr,
            max_length=args.max_length, seed=args.seed,
            label_extractor_train=pseudo_label,
        )
        (stage2_dir / "results.json").write_text(json.dumps(
            {"val_metrics": val_m2, "test_metrics": test_m2,
             "config": {"model_name": args.model_name, "phase": "stage_2_pure_ST",
                        "n_clean": N, "n_pseudo_train": len(pseudo_rows),
                        "lora_r": args.lora_r}}, indent=2))
        print(f"  [stage_2] val_macF1={val_m2.get('f1_macro'):.4f} "
              f"test_macF1={test_m2.get('f1_macro'):.4f}")

        # Stage 3 (CFT): continue model_2 with N golden labels
        print(f"\n--- Stage 3 (CFT): continue training on {len(clean_rows)} golden labels ---")
        stage2_cft_dir = args.out_dir / "stage_2_cft"
        val_m3, test_m3 = train_eval(
            model_2, tok_2, clean_rows, val_rows, test_rows,
            out_dir=stage2_cft_dir, num_labels=args.num_labels,
            num_epochs=args.cft_epochs, batch_size=args.batch_size,
            grad_accum=args.grad_accum, lr=args.cft_lr,
            max_length=args.max_length, seed=args.seed,
            label_key_train="label",
        )
        (stage2_cft_dir / "results.json").write_text(json.dumps(
            {"val_metrics": val_m3, "test_metrics": test_m3,
             "config": {"model_name": args.model_name,
                        "phase": "stage_2_cft_canonical_ST",
                        "n_clean": N, "ws_lr": args.ws_lr, "cft_lr": args.cft_lr,
                        "lora_r": args.lora_r}}, indent=2))
        print(f"  [stage_2_cft] val_macF1={val_m3.get('f1_macro'):.4f} "
              f"test_macF1={test_m3.get('f1_macro'):.4f}")
        del model_2, tok_2
        torch.cuda.empty_cache()

        (args.out_dir / "self_training_summary.json").write_text(json.dumps({
            "n_clean": N,
            "stage_1_golden": {"val": val_m1, "test": test_m1},
            "stage_2_pure_ST": {"val": val_m2, "test": test_m2},
            "stage_2_cft_canonical_ST": {"val": val_m3, "test": test_m3},
        }, indent=2))
        print(f"\nSelf-Training summary -> {args.out_dir/'self_training_summary.json'}")


if __name__ == "__main__":
    main()
