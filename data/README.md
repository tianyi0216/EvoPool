# EvoPool data preparation

This directory ships three dataset prep scripts plus an optional FEVER
metadata enrichment pass. Each script consumes a small piece of raw data
(which the user downloads first) and emits the canonical EvoPool layout:

```
data/processed/<dataset>/
    train.jsonl     # one JSON record per line
    val.jsonl       # capped to val_budget (default 500)
    test.jsonl
    label_map.json  # {class_id_str: class_name}
    dataset.json    # task_type / task_family / counts / label distribution
```

The top-level dispatcher reads ``config.yaml`` and runs the right script for
``dataset.name`` (``chemprot`` / ``fever`` / ``pubmed``):

```bash
python -m data.prepare --config config.yaml
# or via the launcher (same effect):
bash process_data.sh
```

---

## Supported datasets

### 1. ChemProt (10-class biomedical relation extraction)

Source: WRENCH weak-supervision benchmark.

- Download the WRENCH ChemProt files from
  <https://github.com/JieyuZ2/wrench/tree/main/datasets/chemprot>
  and place ``train.json``, ``valid.json``, ``test.json`` under your chosen
  ``dataset.raw_dir`` (default ``data/raw/chemprot/``).
- Run:
  ```bash
  python -m data.prepare_chemprot \
      --raw_dir data/raw/chemprot \
      --out_dir data/processed/chemprot
  ```

### 2. FEVER (3-class fact verification)

Source: HuggingFace ``copenlu/fever_gold_evidence``.

- No manual download needed; the script loads from HuggingFace on first run.
- Default stratified subsample: 10000 train / 1000 val / 7600 test (seed 42).
- Run:
  ```bash
  python -m data.prepare_fever --out_dir data/processed/fever
  ```
- Optional 5-feature enrichment (A semantic / B NLI / C dependency parse /
  D antonym / E phrase banks). NLI and dep-parse require ``torch`` /
  ``transformers`` / ``spacy``:
  ```bash
  python -m data.enrich_fever \
      --input_dir  data/processed/fever \
      --output_dir data/processed/fever_enriched \
      --features   A,B,C,D,E
  ```
  Enabling enrichment from config: set ``dataset.fever_enriched: true`` and
  optionally ``dataset.fever_enrichment_features: "A,B,C,D,E"``.

### 3. PubMed multi-label (14-class MeSH)

Source: Kaggle "PubMed Multi Label Text Classification" dataset
(<https://www.kaggle.com/datasets/owaiskhan9654/pubmed-multilabel-text-classification>).

- Download the dataset zip from Kaggle (sign-in required) and note its path,
  e.g. ``data/raw/pubmed/archive.zip``.
- Run:
  ```bash
  python -m data.prepare_pubmed \
      --kaggle_pubmed_zip data/raw/pubmed/archive.zip \
      --out_dir data/processed/pubmed
  ```
- Default split: 80 % train / 10 % val (capped to ``val_budget``) / remainder test.

---

## Schema

### Single-label tasks (ChemProt, FEVER)

```json
{
  "id": "chemprot_train_42",
  "text": "...",
  "true_label": 3,
  "true_label_name": "Downregulator",
  "label": 3,
  "label_name": "Downregulator",
  "metadata": {
    "source": "chemprot",
    "original_split": "train",
    "original_idx": 42,
    "text_normalized": "...",
    ...
  }
}
```

### Multi-label tasks (PubMed)

```json
{
  "id": "pubmed_17",
  "text": "...",
  "true_labels": [0, 4, 9],
  "true_label_names": ["A", "E", "J"],
  "metadata": {"source": "kaggle:archive.zip", "num_chars": 1234, "num_words": 198}
}
```

The pipeline picks the right reader based on ``dataset.json["task_type"]``
(``single_label`` vs ``multi_label``).

---

## Validation budget

Every dataset caps its validation split to ``val_budget`` (default 500). This
matches the paper convention and keeps the EvoAgg LR-CV step lightweight.
