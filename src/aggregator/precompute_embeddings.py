"""EvoPool: Precompute sentence-BERT text embeddings for EvoAgg.

Caches dense per-example embeddings to a single .npz file shared by all later
EvoAgg invocations. Run once per task; subsequent EvoAgg fits load the cache
instead of re-encoding (a ChemProt train+val+test pass takes ~2-8 minutes CPU
or ~30s on a GPU).

CLI:
    python -m src.aggregator.precompute_embeddings \\
        --task chemprot \\
        --data_dir data/processed/chemprot \\
        --out_path data/embeddings/chemprot_minilm_l6_v2.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from src.utils.eval import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="Task name (used for default out_path)")
    ap.add_argument("--data_dir", type=Path, default=None,
                    help="Directory containing {train,val,test}.jsonl. "
                         "Defaults to data/processed/<task>.")
    ap.add_argument("--out_path", type=Path, default=None,
                    help="Output .npz path. Defaults to "
                         "data/embeddings/<task>_minilm_l6_v2.npz.")
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model id.")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    data_dir = args.data_dir or Path(f"data/processed/{args.task}")
    out_path = args.out_path or Path(f"data/embeddings/{args.task}_minilm_l6_v2.npz")

    from sentence_transformers import SentenceTransformer

    print(f"[precompute] task={args.task} model={args.model_name}")
    print(f"[precompute] reading from {data_dir}")
    model = SentenceTransformer(args.model_name)

    all_ids: List[str] = []
    all_embs: List[np.ndarray] = []
    all_splits: List[str] = []
    for split in ("train", "val", "test"):
        p = data_dir / f"{split}.jsonl"
        if not p.exists():
            print(f"[precompute] WARNING split missing: {p}; skipping")
            continue
        rows = read_jsonl(p)
        texts = [str(r.get("text", "")) for r in rows]
        ids = [str(r.get("id", i)) for i, r in enumerate(rows)]
        print(f"[precompute] encoding {split}: {len(texts)} texts")
        embs = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        all_ids.extend(ids)
        all_embs.append(embs.astype(np.float32))
        all_splits.extend([split] * len(ids))

    if not all_embs:
        raise SystemExit(f"[precompute] no splits found under {data_dir}")

    arr = np.concatenate(all_embs, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ids=np.array(all_ids, dtype=object),
        embs=arr,
        splits=np.array(all_splits, dtype=object),
        model=np.array([args.model_name], dtype=object),
    )
    print(f"[precompute] wrote {out_path}  shape={arr.shape}")


if __name__ == "__main__":
    main()
