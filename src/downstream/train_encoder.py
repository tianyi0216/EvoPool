"""EvoPool: single-label downstream training (RoBERTa-large full FT).

Trains a downstream classifier on EvoPool-generated annotator pseudo-labels and
optionally continues fine-tuning on a small clean (golden) split — the CFT label-
efficiency curve.

Conditions produced (when --clean_sizes is set):
  - golden_clean_<N>   : supervised on N clean labels (no pseudo-label warmup)
  - ours_iter_<II>     : warm-up on the iteration II annotator-aggregated labels
  - cft_clean_<N>_iter_<II> : warm-up on noisy labels, then fine-tune on N clean
  - oracle_all_true    : upper bound (all true labels)

Input: aggregated labels at ``<run_root>/iter_XX/eval/train_labeled.jsonl``.
Optionally an EvoAgg-produced hard-label JSONL via ``--hard_label_path``.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report

from src.downstream.metrics import compute_metrics
from src.utils.io import read_jsonl, write_json


# ── helpers ────────────────────────────────────────────────────────────


def extract_labeled_rows(
    rows: List[Dict[str, Any]],
    aggregation: str = "majority_vote",
    min_depth: int = 0,
    min_agreement: float = 0.0,
    confidence_weighting: bool = False,
    return_ids: bool = False,
    override_labels_by_id: Optional[Dict[str, int]] = None,
):
    """Extract (texts, labels, weights[, ids]) from aggregated annotator rows."""
    import math
    texts, labels, weights, ids = [], [], [], []
    n_abstain, n_depth, n_agree = 0, 0, 0
    for r in rows:
        if override_labels_by_id is not None:
            lab = override_labels_by_id.get(str(r.get("id")), -1)
        else:
            lab = r.get("aggregated_labels", {}).get(aggregation, -1)
        if lab == -1:
            n_abstain += 1
            continue

        votes = r.get("votes", {})
        active = [v for v in votes.values() if v != -1]
        depth = len(active)

        if depth < min_depth:
            n_depth += 1
            continue

        if min_agreement > 0 and depth > 0:
            cnt = Counter(active)
            agreement = cnt.most_common(1)[0][1] / depth
            if agreement < min_agreement:
                n_agree += 1
                continue

        texts.append(r["text"])
        labels.append(int(lab))
        ids.append(r.get("id"))

        if confidence_weighting and depth > 0:
            cnt = Counter(active)
            agreement = cnt.most_common(1)[0][1] / depth
            weights.append(agreement * math.log2(1 + depth))
        else:
            weights.append(1.0)

    total = len(rows)
    print(f"  Labels: {len(texts)}/{total} kept "
          f"(abstain={n_abstain}, depth<{min_depth}={n_depth}, "
          f"agree<{min_agreement:.0%}={n_agree})")
    if return_ids:
        return texts, labels, weights, ids
    return texts, labels, weights


def stratified_subsample(texts: List[str], labels: List[int], n: int,
                         seed: int) -> Tuple[List[str], List[int]]:
    """Stratified subsample preserving per-class proportions."""
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        by_class.setdefault(lab, []).append(i)

    selected: List[int] = []
    total = len(labels)
    for cls, indices in by_class.items():
        k = max(1, round(len(indices) / total * n))
        k = min(k, len(indices))
        selected.extend(rng.sample(indices, k))

    if len(selected) < n:
        remaining = [i for i in range(total) if i not in set(selected)]
        extra = rng.sample(remaining, min(n - len(selected), len(remaining)))
        selected.extend(extra)
    elif len(selected) > n:
        selected = rng.sample(selected, n)

    rng.shuffle(selected)
    return [texts[i] for i in selected], [labels[i] for i in selected]


def load_hard_override(path: Path) -> Dict[str, int]:
    """Load an {id -> argmax-label} map from an EvoAgg / aggregator dump.

    Accepts either ``{"id":..., "soft_label": int}`` or
    ``{"id":..., "label": int}``.
    """
    out: Dict[str, int] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            rid = r.get("id")
            lab = r.get("soft_label")
            if lab is None:
                lab = r.get("label")
            if rid is None or lab is None:
                continue
            out[str(rid)] = int(lab)
    return out


def _resolve_iter_path(template: Optional[Path], iter_idx: int) -> Optional[Path]:
    if template is None:
        return None
    s = str(template)
    if "{iter" in s:
        s = s.format(iter=iter_idx)
    return Path(s)


def load_soft_labels(soft_label_path: Path, texts: List[str],
                     hard_labels: List[int], num_classes: int,
                     row_ids: Optional[List[Any]] = None) -> Tuple[np.ndarray, int]:
    """Match a proba JSONL (id -> proba vector) to training texts."""
    proba_rows = []
    with soft_label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                proba_rows.append(json.loads(line))
    by_id = {r.get("id"): r for r in proba_rows if r.get("id") is not None}

    n_matched = 0
    soft = np.zeros((len(texts), num_classes), dtype=np.float32)
    if row_ids is not None:
        for i, rid in enumerate(row_ids):
            r = by_id.get(rid)
            if r is None or not r.get("proba"):
                soft[i, int(hard_labels[i])] = 1.0
                continue
            arr = np.asarray(r["proba"], dtype=np.float32)
            if arr.shape[0] < num_classes:
                pad = np.full((num_classes - arr.shape[0],), 1e-6, dtype=np.float32)
                arr = np.concatenate([arr, pad])
            elif arr.shape[0] > num_classes:
                arr = arr[:num_classes]
            soft[i] = arr
            s = soft[i].sum()
            if s > 0:
                soft[i] /= s
            else:
                soft[i, int(hard_labels[i])] = 1.0
            n_matched += 1
    else:
        for i, lab in enumerate(hard_labels):
            soft[i, int(lab)] = 1.0
    return soft, n_matched


# ── datasets / trainers ────────────────────────────────────────────────


class TextDataset(Dataset):
    def __init__(self, encodings, labels, sample_weights=None):
        self.encodings = encodings
        self.labels = labels
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(
                self.sample_weights[idx], dtype=torch.float32)
        return item


class WeightedTrainer(Trainer):
    """HF Trainer that applies per-sample weights to the cross-entropy loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weight = inputs.pop("sample_weight", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if sample_weight is not None and labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_sample_loss = loss_fct(
                logits.view(-1, model.config.num_labels), labels.view(-1))
            loss = (per_sample_loss * sample_weight.view(-1)).mean()
        else:
            loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss


class SoftTextDataset(Dataset):
    """Dataset that carries per-example probability vectors for soft-CE training."""

    def __init__(self, encodings, soft_targets, hard_labels):
        self.encodings = encodings
        self.soft_targets = soft_targets
        self.hard_labels = hard_labels

    def __len__(self):
        return len(self.hard_labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["soft_targets"] = torch.tensor(self.soft_targets[idx], dtype=torch.float32)
        item["labels"] = torch.tensor(self.hard_labels[idx], dtype=torch.long)
        return item


class SoftLabelTrainer(Trainer):
    """Trainer that minimises soft cross-entropy (distillation loss)."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        soft_targets = inputs.pop("soft_targets", None)
        if soft_targets is None:
            outputs = model(**inputs)
            loss = outputs.get("loss")
            return (loss, outputs) if return_outputs else loss
        inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return (loss, outputs) if return_outputs else loss


# ── core training ──────────────────────────────────────────────────────


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(model_name: str, num_labels: int, from_scratch: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if from_scratch:
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels,
        )
    return tokenizer, model


def train_and_evaluate(
    train_texts: List[str], train_labels: List[int],
    val_texts: List[str], val_labels: List[int],
    test_texts: List[str], test_labels: List[int],
    model_name: str,
    out_dir: Path,
    seed: int = 42,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    early_stopping_patience: int = 3,
    from_scratch: bool = False,
    sample_weights: Optional[List[float]] = None,
    soft_targets: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    _set_seed(seed)

    num_labels = len(set(train_labels) | set(val_labels) | set(test_labels))
    if soft_targets is not None:
        num_labels = max(num_labels, soft_targets.shape[1])
    tokenizer, model = _build_model(model_name, num_labels, from_scratch)

    train_enc = tokenizer(train_texts, truncation=True, padding=True,
                          max_length=max_length, return_tensors="pt")
    val_enc = tokenizer(val_texts, truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt")
    test_enc = tokenizer(test_texts, truncation=True, padding=True,
                         max_length=max_length, return_tensors="pt")

    use_soft = soft_targets is not None
    use_weighted = (sample_weights is not None) and not use_soft
    if use_soft:
        train_ds = SoftTextDataset(train_enc, soft_targets, train_labels)
    else:
        train_ds = TextDataset(train_enc, train_labels, sample_weights)
    val_ds = TextDataset(val_enc, val_labels)
    test_ds = TextDataset(test_enc, test_labels)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=not use_soft,
    )
    if use_soft:
        trainer_cls = SoftLabelTrainer
    elif use_weighted:
        trainer_cls = WeightedTrainer
    else:
        trainer_cls = Trainer
    trainer = trainer_cls(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    train_result = trainer.train()
    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    test_pred = trainer.predict(test_ds)
    test_preds = np.argmax(test_pred.predictions, axis=-1)
    all_labels = sorted(set(test_labels) | set(train_labels))
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {}
    target_names = [id2label.get(i, str(i)) for i in all_labels]
    report = classification_report(test_labels, test_preds, labels=all_labels,
                                   target_names=target_names, digits=4,
                                   zero_division=0)

    results = {
        "model_name": model_name,
        "from_scratch": from_scratch,
        "soft_targets": bool(use_soft),
        "num_train": len(train_texts),
        "train_label_dist": dict(Counter(train_labels)),
        "train_loss": train_result.metrics.get("train_loss"),
        "epochs_trained": train_result.metrics.get("epoch"),
        "val_metrics": {k.replace("eval_", ""): v for k, v in val_metrics.items()},
        "test_metrics": {k.replace("test_", ""): v for k, v in test_metrics.items()},
        "classification_report": report,
    }
    write_json(out_dir / "results.json", results)
    (out_dir / "classification_report.txt").write_text(report + "\n")
    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))

    ckpt_dir = out_dir / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    return results


def train_cft(
    stage1_texts: List[str], stage1_labels: List[int],
    cft_texts: List[str], cft_labels: List[int],
    val_texts: List[str], val_labels: List[int],
    test_texts: List[str], test_labels: List[int],
    model_name: str, out_dir: Path,
    seed: int = 42,
    num_epochs: int = 10, cft_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5, cft_learning_rate: float = 1e-5,
    max_length: int = 128,
    early_stopping_patience: int = 3,
    from_scratch: bool = False,
    stage1_soft_targets: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Two-stage CFT: warm-up on noisy labels, then fine-tune on N clean labels."""
    _set_seed(seed)
    num_labels = len(set(stage1_labels) | set(cft_labels) | set(val_labels) | set(test_labels))
    if stage1_soft_targets is not None:
        num_labels = max(num_labels, stage1_soft_targets.shape[1])
    tokenizer, model = _build_model(model_name, num_labels, from_scratch)

    s1_enc = tokenizer(stage1_texts, truncation=True, padding=True,
                       max_length=max_length, return_tensors="pt")
    val_enc = tokenizer(val_texts, truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt")
    s1_use_soft = stage1_soft_targets is not None
    s1_ds = (SoftTextDataset(s1_enc, stage1_soft_targets, stage1_labels)
             if s1_use_soft else TextDataset(s1_enc, stage1_labels))
    val_ds = TextDataset(val_enc, val_labels)

    s1_args = TrainingArguments(
        output_dir=str(out_dir / "stage1_ckpts"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=learning_rate,
        warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro", greater_is_better=True,
        save_total_limit=2, logging_steps=50, report_to="none",
        seed=seed, fp16=torch.cuda.is_available(),
        remove_unused_columns=not s1_use_soft,
    )
    s1_trainer_cls = SoftLabelTrainer if s1_use_soft else Trainer
    s1_trainer = s1_trainer_cls(
        model=model, args=s1_args,
        train_dataset=s1_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    s1_result = s1_trainer.train()
    s1_ckpt = out_dir / "stage1_ckpts"
    if s1_ckpt.exists():
        shutil.rmtree(s1_ckpt)

    # Stage 2: fine-tune on N clean labels with 80/20 train/early-stop split.
    rng = random.Random(seed)
    idxs = list(range(len(cft_texts)))
    rng.shuffle(idxs)
    split = max(1, int(0.8 * len(idxs)))
    s2_train_idx = idxs[:split]
    s2_es_idx = idxs[split:] if split < len(idxs) else idxs[:1]
    s2_train_texts = [cft_texts[i] for i in s2_train_idx]
    s2_train_labels = [cft_labels[i] for i in s2_train_idx]
    s2_es_texts = [cft_texts[i] for i in s2_es_idx]
    s2_es_labels = [cft_labels[i] for i in s2_es_idx]

    s2_train_enc = tokenizer(s2_train_texts, truncation=True, padding=True,
                             max_length=max_length, return_tensors="pt")
    s2_es_enc = tokenizer(s2_es_texts, truncation=True, padding=True,
                          max_length=max_length, return_tensors="pt")
    s2_train_ds = TextDataset(s2_train_enc, s2_train_labels)
    s2_es_ds = TextDataset(s2_es_enc, s2_es_labels)

    s2_args = TrainingArguments(
        output_dir=str(out_dir / "stage2_ckpts"),
        num_train_epochs=cft_epochs,
        per_device_train_batch_size=min(batch_size, max(4, len(s2_train_texts) // 4)),
        per_device_eval_batch_size=64,
        learning_rate=cft_learning_rate,
        warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro", greater_is_better=True,
        save_total_limit=2, logging_steps=10, report_to="none",
        seed=seed, fp16=torch.cuda.is_available(),
    )
    s2_trainer = Trainer(
        model=model, args=s2_args,
        train_dataset=s2_train_ds, eval_dataset=s2_es_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    s2_result = s2_trainer.train()

    test_enc = tokenizer(test_texts, truncation=True, padding=True,
                         max_length=max_length, return_tensors="pt")
    test_ds = TextDataset(test_enc, test_labels)
    val_metrics = s2_trainer.evaluate(val_ds)
    test_metrics = s2_trainer.evaluate(test_ds, metric_key_prefix="test")
    test_pred = s2_trainer.predict(test_ds)
    test_preds = np.argmax(test_pred.predictions, axis=-1)
    all_labels = sorted(set(test_labels) | set(stage1_labels))
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {}
    target_names = [id2label.get(i, str(i)) for i in all_labels]
    report = classification_report(test_labels, test_preds, labels=all_labels,
                                   target_names=target_names, digits=4,
                                   zero_division=0)

    results = {
        "model_name": model_name,
        "from_scratch": from_scratch,
        "soft_targets_stage1": bool(s1_use_soft),
        "cft": True,
        "num_stage1_train": len(stage1_texts),
        "num_cft_train": len(s2_train_texts),
        "num_train": len(stage1_texts),
        "train_label_dist": dict(Counter(stage1_labels)),
        "stage1_loss": s1_result.metrics.get("train_loss"),
        "stage2_loss": s2_result.metrics.get("train_loss"),
        "val_metrics": {k.replace("eval_", ""): v for k, v in val_metrics.items()},
        "test_metrics": {k.replace("test_", ""): v for k, v in test_metrics.items()},
        "classification_report": report,
    }
    write_json(out_dir / "results.json", results)
    (out_dir / "classification_report.txt").write_text(report + "\n")
    s2_trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))
    s2_ckpt = out_dir / "stage2_ckpts"
    if s2_ckpt.exists():
        shutil.rmtree(s2_ckpt)
    return results


# ── main ───────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", type=Path, required=True,
                   help="EvoPool run root (contains iter_00/, iter_01/, ...)")
    p.add_argument("--train_path", type=Path, required=True,
                   help="Raw training data with true labels")
    p.add_argument("--val_path", type=Path, required=True)
    p.add_argument("--test_path", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--model_name", type=str, default="FacebookAI/roberta-large")
    p.add_argument("--aggregation", type=str, default="majority_vote")
    p.add_argument("--label_fractions", type=str, default="0.2,0.25,0.33,0.5,1.0")
    p.add_argument("--iterations", type=str, default="0,1,2,5",
                   help="Iteration indices to train on (comma-separated)")
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--from_scratch", action="store_true")
    p.add_argument("--cft", action="store_true")
    p.add_argument("--cft_epochs", type=int, default=5)
    p.add_argument("--cft_learning_rate", type=float, default=1e-5)
    p.add_argument("--cft_n_clean", type=int, default=None)
    p.add_argument("--clean_sizes", type=str, default=None,
                   help="WS-style label-efficiency curve sizes, e.g. '25,50,100,200,500,1000'")
    p.add_argument("--best_iteration", type=int, default=None)
    p.add_argument("--min_vote_depth", type=int, default=0)
    p.add_argument("--min_agreement", type=float, default=0.0)
    p.add_argument("--confidence_weight", action="store_true")
    p.add_argument("--soft_label_path", type=Path, default=None)
    p.add_argument("--hard_label_path", type=Path, default=None,
                   help="EvoAgg / aggregator dump that overrides the noisy MV labels")
    p.add_argument("--soft_num_classes", type=int, default=None)
    args = p.parse_args()

    summary_path = args.out_dir / "summary.json"
    if summary_path.exists():
        print(f"[skip] {summary_path} already exists; delete it or pass a fresh --out_dir to rerun.")
        return

    fractions = [float(x) for x in args.label_fractions.split(",")]
    iter_indices = [int(x) for x in args.iterations.split(",")]
    prefix = "scratch_" if args.from_scratch else ""

    val_rows = read_jsonl(args.val_path)
    test_rows = read_jsonl(args.test_path)
    val_texts = [r["text"] for r in val_rows]
    val_labels = [int(r["true_label"]) for r in val_rows]
    test_texts = [r["text"] for r in test_rows]
    test_labels = [int(r["true_label"]) for r in test_rows]

    train_rows = read_jsonl(args.train_path)
    all_train_texts = [r["text"] for r in train_rows]
    all_train_labels = [int(r["true_label"]) for r in train_rows]
    N = len(all_train_texts)

    all_results: List[Dict[str, Any]] = []

    if args.clean_sizes:
        clean_sizes = [int(x) for x in args.clean_sizes.split(",")]
        best_iter = args.best_iteration if args.best_iteration is not None else iter_indices[-1]
        V = len(val_texts)
        print(f"\n*** Label-efficiency curve ***")
        print(f"    clean_sizes = {clean_sizes}")
        print(f"    best_iteration = {best_iter}")

        # Golden baselines: supervised on N clean val examples
        for n_clean in clean_sizes:
            n_clean = min(n_clean, V)
            tag = f"{prefix}golden_clean_{n_clean}"
            out = args.out_dir / tag
            if n_clean >= V:
                split = int(0.8 * V)
                rng = random.Random(args.seed)
                idxs = list(range(V))
                rng.shuffle(idxs)
                g_train_idx, g_es_idx = idxs[:split], idxs[split:]
                g_train_texts = [val_texts[i] for i in g_train_idx]
                g_train_labels = [val_labels[i] for i in g_train_idx]
                g_es_texts = [val_texts[i] for i in g_es_idx]
                g_es_labels = [val_labels[i] for i in g_es_idx]
            else:
                rng = random.Random(args.seed)
                by_class: Dict[int, List[int]] = {}
                for i, lab in enumerate(val_labels):
                    by_class.setdefault(lab, []).append(i)
                selected: List[int] = []
                for cls, idxs in by_class.items():
                    k = max(1, round(len(idxs) / V * n_clean))
                    k = min(k, len(idxs))
                    selected.extend(rng.sample(idxs, k))
                if len(selected) < n_clean:
                    remaining = [i for i in range(V) if i not in set(selected)]
                    extra = rng.sample(remaining, min(n_clean - len(selected), len(remaining)))
                    selected.extend(extra)
                elif len(selected) > n_clean:
                    selected = rng.sample(selected, n_clean)
                sel_set = set(selected)
                g_train_texts = [val_texts[i] for i in selected]
                g_train_labels = [val_labels[i] for i in selected]
                rest = [i for i in range(V) if i not in sel_set]
                if len(rest) < 4:
                    g_es_texts, g_es_labels = g_train_texts, g_train_labels
                else:
                    g_es_texts = [val_texts[i] for i in rest]
                    g_es_labels = [val_labels[i] for i in rest]
            print(f"\n{'='*60}\nCONDITION: {tag} (train={len(g_train_texts)}, es={len(g_es_texts)})\n{'='*60}")
            result = train_and_evaluate(
                g_train_texts, g_train_labels, g_es_texts, g_es_labels,
                test_texts, test_labels, model_name=args.model_name, out_dir=out,
                seed=args.seed, num_epochs=args.num_epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate, max_length=args.max_length,
                early_stopping_patience=args.early_stopping_patience,
                from_scratch=args.from_scratch,
            )
            result["condition"] = tag
            result["condition_type"] = "golden_clean"
            result["n_clean"] = n_clean
            all_results.append(result)

        # Ours: noisy labels only at best_iter
        iter_dir = args.run_root / f"iter_{best_iter:02d}"
        labeled_path = iter_dir / "eval" / "train_labeled.jsonl"
        if labeled_path.exists():
            rows = read_jsonl(labeled_path)
            _hard_override = None
            if args.hard_label_path:
                _hp = _resolve_iter_path(args.hard_label_path, best_iter)
                if _hp and _hp.exists():
                    _hard_override = load_hard_override(_hp)
                    print(f"  [hard-label override] loaded {len(_hard_override)} labels from {_hp}")
            noisy_texts, noisy_labels, noisy_w, noisy_ids = extract_labeled_rows(
                rows, args.aggregation, args.min_vote_depth, args.min_agreement,
                args.confidence_weight, return_ids=True,
                override_labels_by_id=_hard_override,
            )
            if noisy_texts:
                tag = f"{prefix}ours_iter_{best_iter:02d}"
                out = args.out_dir / tag
                print(f"\n{'='*60}\nCONDITION: {tag} (n={len(noisy_texts)}, noisy labels)\n{'='*60}")
                sw = noisy_w if args.confidence_weight else None
                soft = None
                soft_path = _resolve_iter_path(args.soft_label_path, best_iter)
                if soft_path is not None and soft_path.exists():
                    nc = args.soft_num_classes or len(set(noisy_labels) | set(val_labels) | set(test_labels))
                    soft, n_matched = load_soft_labels(soft_path, noisy_texts,
                                                       noisy_labels, nc, row_ids=noisy_ids)
                    print(f"  Soft labels: {n_matched}/{len(noisy_texts)} matched")
                result = train_and_evaluate(
                    noisy_texts, noisy_labels, val_texts, val_labels,
                    test_texts, test_labels, model_name=args.model_name, out_dir=out,
                    seed=args.seed, num_epochs=args.num_epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, max_length=args.max_length,
                    early_stopping_patience=args.early_stopping_patience,
                    from_scratch=args.from_scratch,
                    sample_weights=sw, soft_targets=soft,
                )
                result["condition"] = tag
                result["condition_type"] = "ours"
                result["iteration"] = best_iter
                all_results.append(result)

                # Ours + CFT for each clean size
                for n_clean in clean_sizes:
                    n_clean = min(n_clean, V)
                    if n_clean < V:
                        cft_texts, cft_labels = stratified_subsample(
                            val_texts, val_labels, n_clean, args.seed)
                    else:
                        cft_texts, cft_labels = val_texts, val_labels
                    tag = f"{prefix}cft_clean_{n_clean}_iter_{best_iter:02d}"
                    out = args.out_dir / tag
                    print(f"\n{'='*60}\nCONDITION: {tag} (stage1={len(noisy_texts)} noisy -> stage2={n_clean} clean)\n{'='*60}")
                    result = train_cft(
                        stage1_texts=noisy_texts, stage1_labels=noisy_labels,
                        cft_texts=cft_texts, cft_labels=cft_labels,
                        val_texts=val_texts, val_labels=val_labels,
                        test_texts=test_texts, test_labels=test_labels,
                        model_name=args.model_name, out_dir=out, seed=args.seed,
                        num_epochs=args.num_epochs, cft_epochs=args.cft_epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        cft_learning_rate=args.cft_learning_rate,
                        max_length=args.max_length,
                        early_stopping_patience=args.early_stopping_patience,
                        from_scratch=args.from_scratch,
                        stage1_soft_targets=soft,
                    )
                    result["condition"] = tag
                    result["condition_type"] = "ours_cft"
                    result["iteration"] = best_iter
                    result["n_clean"] = n_clean
                    all_results.append(result)
        else:
            print(f"WARNING: {labeled_path} not found; skipping ours/CFT conditions")

        # Oracle (all true labels)
        tag = f"{prefix}oracle_all_true"
        out = args.out_dir / tag
        print(f"\n{'='*60}\nCONDITION: {tag} (n={N})\n{'='*60}")
        result = train_and_evaluate(
            all_train_texts, all_train_labels, val_texts, val_labels,
            test_texts, test_labels, model_name=args.model_name, out_dir=out,
            seed=args.seed, num_epochs=args.num_epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, max_length=args.max_length,
            early_stopping_patience=args.early_stopping_patience,
            from_scratch=args.from_scratch,
        )
        result["condition"] = tag
        result["condition_type"] = "oracle"
        all_results.append(result)

    else:
        # True-label fraction baselines
        for frac in fractions:
            n = max(10, round(N * frac))
            n = min(n, N)
            tag = f"{prefix}true_frac_{frac:.2f}"
            sub_texts, sub_labels = stratified_subsample(
                all_train_texts, all_train_labels, n, args.seed)
            out = args.out_dir / tag
            print(f"\n{'='*60}\nCONDITION: {tag} (n={len(sub_texts)})\n{'='*60}")
            result = train_and_evaluate(
                sub_texts, sub_labels, val_texts, val_labels,
                test_texts, test_labels, model_name=args.model_name, out_dir=out,
                seed=args.seed, num_epochs=args.num_epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate, max_length=args.max_length,
                early_stopping_patience=args.early_stopping_patience,
                from_scratch=args.from_scratch,
            )
            result["condition"] = tag
            result["condition_type"] = "true_labels"
            result["label_fraction"] = frac
            all_results.append(result)

        # Ours per iteration
        for it in iter_indices:
            iter_dir = args.run_root / f"iter_{it:02d}"
            labeled_path = iter_dir / "eval" / "train_labeled.jsonl"
            if not labeled_path.exists():
                print(f"SKIP iter_{it:02d}: {labeled_path} not found")
                continue
            rows = read_jsonl(labeled_path)
            _hard_override = None
            if args.hard_label_path:
                _hp = _resolve_iter_path(args.hard_label_path, it)
                if _hp and _hp.exists():
                    _hard_override = load_hard_override(_hp)
            texts, labels, sw, row_ids = extract_labeled_rows(
                rows, args.aggregation, args.min_vote_depth, args.min_agreement,
                confidence_weighting=args.confidence_weight, return_ids=True,
                override_labels_by_id=_hard_override,
            )
            if not texts:
                print(f"SKIP iter_{it:02d}: 0 labeled examples")
                continue
            tag = f"{prefix}ours_iter_{it:02d}"
            out = args.out_dir / tag
            print(f"\n{'='*60}\nCONDITION: {tag} (n={len(texts)})\n{'='*60}")
            soft = None
            soft_path = _resolve_iter_path(args.soft_label_path, it)
            if soft_path is not None and soft_path.exists():
                nc = args.soft_num_classes or len(set(labels) | set(val_labels) | set(test_labels))
                soft, n_matched = load_soft_labels(soft_path, texts, labels, nc, row_ids=row_ids)
                print(f"  Soft labels: {n_matched}/{len(texts)} matched")
            result = train_and_evaluate(
                texts, labels, val_texts, val_labels, test_texts, test_labels,
                model_name=args.model_name, out_dir=out,
                seed=args.seed, num_epochs=args.num_epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate, max_length=args.max_length,
                early_stopping_patience=args.early_stopping_patience,
                from_scratch=args.from_scratch,
                sample_weights=sw if args.confidence_weight else None,
                soft_targets=soft,
            )
            result["condition"] = tag
            result["condition_type"] = "ours"
            result["iteration"] = it
            all_results.append(result)

        # Ours + CFT
        if args.cft:
            cft_n = args.cft_n_clean or len(val_texts)
            if cft_n < len(val_texts):
                cft_texts, cft_labels = stratified_subsample(
                    val_texts, val_labels, cft_n, args.seed)
            else:
                cft_texts, cft_labels = val_texts, val_labels
            for it in iter_indices:
                iter_dir = args.run_root / f"iter_{it:02d}"
                labeled_path = iter_dir / "eval" / "train_labeled.jsonl"
                if not labeled_path.exists():
                    continue
                rows = read_jsonl(labeled_path)
                texts, labels, _ = extract_labeled_rows(
                    rows, args.aggregation, args.min_vote_depth,
                    args.min_agreement, confidence_weighting=False,
                )
                if not texts:
                    continue
                tag = f"{prefix}cft_iter_{it:02d}"
                out = args.out_dir / tag
                print(f"\n{'='*60}\nCONDITION: {tag} (stage1={len(texts)} -> stage2={len(cft_texts)} clean)\n{'='*60}")
                result = train_cft(
                    stage1_texts=texts, stage1_labels=labels,
                    cft_texts=cft_texts, cft_labels=cft_labels,
                    val_texts=val_texts, val_labels=val_labels,
                    test_texts=test_texts, test_labels=test_labels,
                    model_name=args.model_name, out_dir=out, seed=args.seed,
                    num_epochs=args.num_epochs, cft_epochs=args.cft_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    cft_learning_rate=args.cft_learning_rate,
                    max_length=args.max_length,
                    early_stopping_patience=args.early_stopping_patience,
                    from_scratch=args.from_scratch,
                )
                result["condition"] = tag
                result["condition_type"] = "ours_cft"
                result["iteration"] = it
                all_results.append(result)

        # Oracle
        tag = f"{prefix}oracle_all_true"
        out = args.out_dir / tag
        print(f"\n{'='*60}\nCONDITION: {tag} (n={N})\n{'='*60}")
        result = train_and_evaluate(
            all_train_texts, all_train_labels, val_texts, val_labels,
            test_texts, test_labels, model_name=args.model_name, out_dir=out,
            seed=args.seed, num_epochs=args.num_epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, max_length=args.max_length,
            early_stopping_patience=args.early_stopping_patience,
            from_scratch=args.from_scratch,
        )
        result["condition"] = tag
        result["condition_type"] = "oracle"
        all_results.append(result)

    # Summary
    summary = {
        "model_name": args.model_name,
        "aggregation": args.aggregation,
        "from_scratch": args.from_scratch,
        "cft": args.cft if not args.clean_sizes else True,
        "clean_sizes_mode": args.clean_sizes is not None,
        "N_total": N,
        "seed": args.seed,
        "conditions": [],
    }
    for r in all_results:
        entry = {
            "condition": r["condition"],
            "condition_type": r["condition_type"],
            "num_train": r["num_train"],
            "train_label_dist": r["train_label_dist"],
            "test_accuracy": r["test_metrics"].get("accuracy"),
            "test_f1_macro": r["test_metrics"].get("f1_macro"),
            "val_f1_macro": r["val_metrics"].get("f1_macro"),
            "test_metrics_full": r["test_metrics"],
        }
        if "n_clean" in r:
            entry["n_clean"] = r["n_clean"]
        if "iteration" in r:
            entry["iteration"] = r["iteration"]
        summary["conditions"].append(entry)
    write_json(summary_path, summary)

    # Print table
    print(f"\n{'='*70}")
    print(f"Label Efficiency Summary ({args.model_name}{' [scratch]' if args.from_scratch else ''})")
    print(f"{'='*70}")
    print("{:<40} {:<8} {:<10} {:<10} {:<10}".format(
        "Condition", "N_train", "Acc", "F1_macro", "Val_F1"))
    print("-" * 80)
    for c in summary["conditions"]:
        print("{:<40} {:<8} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            c["condition"], c["num_train"],
            c["test_accuracy"] or 0.0, c["test_f1_macro"] or 0.0,
            c["val_f1_macro"] or 0.0))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
