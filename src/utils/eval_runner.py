"""EvoPool: unified eval runner.

Two modes:

1. ``annotator`` — eval the annotator pool produced by the pipeline. Reads the
   latest ``iter_XX/eval/{split}_labeled.jsonl`` under a run directory and
   recomputes macro-F1, accuracy, and coverage for MV / EvoAgg aggregators.

2. ``ckpt`` — eval a saved downstream HuggingFace checkpoint on the dataset's
   test split. Loads via ``AutoModelForSequenceClassification`` and reports
   macro-F1 plus a per-class breakdown.

Usage:
    python -m src.utils.eval_runner --config config.yaml --mode annotator \\
        [--run_dir runs/chemprot_l0] [--evoagg_dir runs/chemprot_l0/evoagg_labels]

    python -m src.utils.eval_runner --config config.yaml --mode ckpt \\
        --ckpt_dir runs/chemprot_l0/downstream/ws/model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.config_loader import get, load_config
from src.utils.eval import ABSTAIN, read_jsonl


def _resolve_latest_iter(run_dir: Path) -> Path:
    iter_dirs = sorted(
        d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")
    )
    if not iter_dirs:
        nested = run_dir / "pipeline"
        if nested.exists():
            iter_dirs = sorted(
                d for d in nested.iterdir() if d.is_dir() and d.name.startswith("iter_")
            )
    if not iter_dirs:
        raise SystemExit(f"No iter_* directory under {run_dir}")
    return iter_dirs[-1]


def _macro_f1_singlelabel(preds: List[int], labels: List[int], n_classes: int) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(labels, preds, labels=list(range(n_classes)),
                          average="macro", zero_division=0))


def _macro_f1_multilabel(preds, labels) -> float:
    from sklearn.metrics import f1_score
    import numpy as np
    return float(f1_score(np.asarray(labels), np.asarray(preds),
                          average="macro", zero_division=0))


def eval_annotator(run_dir: Path, n_classes: int, multi_label: bool,
                   evoagg_dir: Optional[Path] = None) -> Dict[str, Any]:
    iter_dir = _resolve_latest_iter(Path(run_dir))
    eval_dir = iter_dir / "eval"
    test_labeled = eval_dir / "test_labeled.jsonl"
    if not test_labeled.exists():
        raise SystemExit(f"Missing {test_labeled}")

    print(f"[eval-annotator] iter:  {iter_dir.name}")
    print(f"[eval-annotator] file:  {test_labeled}")

    rows = read_jsonl(test_labeled)
    print(f"[eval-annotator] rows:  {len(rows)}")

    report_path = eval_dir / "report.json"
    pool_size = "unknown"
    if report_path.exists():
        try:
            pool_size = json.loads(report_path.read_text()).get("num_annotators", "unknown")
        except json.JSONDecodeError:
            pass
    print(f"[eval-annotator] pool:  {pool_size} annotators")

    results: Dict[str, Any] = {"iter": iter_dir.name, "n_rows": len(rows),
                                "pool_size": pool_size, "aggregators": {}}

    aggs = ["majority_vote", "weighted_vote"]
    for agg in aggs:
        preds, labels, covered = [], [], 0
        for r in rows:
            agg_dict = r.get("aggregated_labels", {}) or {}
            p = agg_dict.get(agg, ABSTAIN)
            y = r.get("true_label") if not multi_label else r.get("true_labels")
            if multi_label:
                if y is None:
                    continue
                vec = [0] * n_classes
                for c in y:
                    if 0 <= c < n_classes:
                        vec[c] = 1
                labels.append(vec)
                if not isinstance(p, list):
                    p = []
                pred_vec = [0] * n_classes
                for c in p:
                    if 0 <= c < n_classes:
                        pred_vec[c] = 1
                preds.append(pred_vec)
                if any(pred_vec):
                    covered += 1
            else:
                if y is None:
                    continue
                if p == ABSTAIN or p is None:
                    preds.append(0)
                else:
                    preds.append(int(p))
                    covered += 1
                labels.append(int(y))

        if not labels:
            print(f"  {agg}: no labels")
            continue
        if multi_label:
            f1 = _macro_f1_multilabel(preds, labels)
            acc = None
        else:
            f1 = _macro_f1_singlelabel(preds, labels, n_classes)
            acc = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
        cov = covered / len(rows)
        print(f"  {agg:14s}  macro_f1={f1:.4f}  acc={acc:.4f}" if acc is not None
              else f"  {agg:14s}  macro_f1={f1:.4f}")
        print(f"                  coverage={cov:.4f}")
        results["aggregators"][agg] = {"macro_f1": f1, "accuracy": acc, "coverage": cov}

    if evoagg_dir is not None:
        evo_test = Path(evoagg_dir) / "test_labeled.jsonl"
        if evo_test.exists():
            print(f"[eval-annotator] EvoAgg: {evo_test}")
            evo_rows = read_jsonl(evo_test)
            preds, labels = [], []
            for r in evo_rows:
                agg_dict = r.get("aggregated_labels", {}) or {}
                p = agg_dict.get("evoagg_argmax") or agg_dict.get("a1_argmax")
                y = r.get("true_label") if not multi_label else r.get("true_labels")
                if y is None or p is None:
                    continue
                if multi_label:
                    vec = [0] * n_classes
                    for c in y:
                        if 0 <= c < n_classes:
                            vec[c] = 1
                    labels.append(vec)
                    pv = [0] * n_classes
                    for c in (p if isinstance(p, list) else []):
                        if 0 <= c < n_classes:
                            pv[c] = 1
                    preds.append(pv)
                else:
                    labels.append(int(y))
                    preds.append(int(p))
            if labels:
                f1 = (_macro_f1_multilabel(preds, labels) if multi_label
                      else _macro_f1_singlelabel(preds, labels, n_classes))
                print(f"  {'evoagg':14s}  macro_f1={f1:.4f}  coverage=1.0000")
                results["aggregators"]["evoagg"] = {"macro_f1": f1, "coverage": 1.0}
        else:
            print(f"[eval-annotator] EvoAgg dir not found: {evo_test} (skipping)")

    return results


def eval_ckpt(ckpt_dir: Path, test_path: Path, n_classes: int,
              multi_label: bool, max_length: int = 256, batch_size: int = 32,
              out_path: Optional[Path] = None) -> Dict[str, Any]:
    import torch
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"[eval-ckpt] checkpoint: {ckpt_dir}")
    print(f"[eval-ckpt] test:       {test_path}")

    rows = read_jsonl(test_path)
    print(f"[eval-ckpt] rows:       {len(rows)}")

    tok = AutoTokenizer.from_pretrained(str(ckpt_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval-ckpt] device:     {device}")
    model.to(device)
    model.eval()

    texts = [r.get("text", "") for r in rows]
    if multi_label:
        labels = []
        for r in rows:
            y = r.get("true_labels") or []
            vec = [0] * n_classes
            for c in y:
                if 0 <= c < n_classes:
                    vec[c] = 1
            labels.append(vec)
    else:
        labels = [int(r.get("true_label", 0)) for r in rows]

    preds: List[Any] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tok(batch, return_tensors="pt", truncation=True,
                         padding=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits.cpu()
            if multi_label:
                probs = torch.sigmoid(logits).numpy()
                preds.extend((probs >= 0.5).astype(int).tolist())
            else:
                preds.extend(logits.argmax(dim=-1).tolist())

    if multi_label:
        macro = float(f1_score(labels, preds, average="macro", zero_division=0))
        micro = float(f1_score(labels, preds, average="micro", zero_division=0))
        print(f"  macro_f1={macro:.4f}  micro_f1={micro:.4f}")
        results = {"macro_f1": macro, "micro_f1": micro, "n_rows": len(rows)}
    else:
        macro = float(f1_score(labels, preds, average="macro",
                                labels=list(range(n_classes)), zero_division=0))
        acc = float(accuracy_score(labels, preds))
        print(f"  macro_f1={macro:.4f}  accuracy={acc:.4f}")
        print()
        print(classification_report(labels, preds, labels=list(range(n_classes)),
                                     zero_division=0, digits=4))
        results = {"macro_f1": macro, "accuracy": acc, "n_rows": len(rows)}

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[eval-ckpt] wrote {out_path}")

    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mode", choices=["annotator", "ckpt"], required=True)
    p.add_argument("--run_dir", type=Path, default=None,
                   help="(annotator mode) run root with iter_XX/; defaults to pipeline.out_root")
    p.add_argument("--evoagg_dir", type=Path, default=None,
                   help="(annotator mode) optional EvoAgg dump dir to also eval")
    p.add_argument("--ckpt_dir", type=Path, default=None,
                   help="(ckpt mode) HuggingFace checkpoint directory")
    p.add_argument("--test_path", type=Path, default=None,
                   help="override config dataset test split jsonl")
    p.add_argument("--out_path", type=Path, default=None,
                   help="(ckpt mode) write results.json to this path")
    args = p.parse_args()

    cfg = load_config(args.config)
    from src.tasks.configs import get_task_config

    task_name = get(cfg, "dataset.name")
    if not task_name:
        raise SystemExit("config.dataset.name must be set.")
    task_cfg = get_task_config(task_name)
    n_classes = int(get(cfg, "dataset.n_classes") or getattr(task_cfg, "num_classes", 0))
    multi_label = bool(getattr(task_cfg, "multi_label", False)) or bool(
        get(cfg, "dataset.multi_label", False)
    )

    if args.mode == "annotator":
        run_dir = args.run_dir or Path(get(cfg, "pipeline.out_root") or "")
        if not run_dir.exists():
            raise SystemExit(f"run_dir does not exist: {run_dir}")
        evoagg_dir = args.evoagg_dir or (
            Path(get(cfg, "aggregator.dump_dir"))
            if get(cfg, "aggregator.dump_dir")
            else None
        )
        eval_annotator(run_dir, n_classes, multi_label, evoagg_dir)
    else:
        if args.ckpt_dir is None:
            raise SystemExit("--ckpt_dir required for ckpt mode")
        test_path = args.test_path or (
            Path(get(cfg, "dataset.processed_dir") or
                 f"data/processed/{task_name}") / "test.jsonl"
        )
        if not test_path.exists():
            raise SystemExit(f"test_path does not exist: {test_path}")
        eval_ckpt(args.ckpt_dir, test_path, n_classes, multi_label,
                  out_path=args.out_path)


if __name__ == "__main__":
    main()
