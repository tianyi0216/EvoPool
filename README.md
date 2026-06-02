<div align="center">

  <h3>EvoPool: Evolutionary Programmatic Annotation for Label-Efficient Specialized Supervision</h3>

  <p>
    <a href="https://tianyi0216.github.io/">Tianyi Xu</a><sup>1,2</sup>&nbsp;&middot;&nbsp;
    <a href="https://mercury7353.github.io/Yaolun-Zhang.github.io/">Yaolun Zhang</a><sup>1</sup>&nbsp;&middot;&nbsp;
    <a href="https://yancyou.github.io/">Xuan Ouyang</a><sup>2</sup>&nbsp;&middot;&nbsp;
    <a href="https://huazhengwang.github.io/">Huazheng Wang</a><sup>1&#9993;</sup>
  </p>

  <p>
    <sup>1</sup> Oregon State University &nbsp;&nbsp;
    <sup>2</sup> University of Wisconsin&ndash;Madison
  </p>

  <p>
    <sup>&#9993;</sup> Corresponding author
  </p>

  <p>
    <a href="https://arxiv.org/abs/2606.01617">
      <img src="https://img.shields.io/badge/arXiv-2606.01617-B31B1B?style=flat-square&logo=arxiv" alt="arXiv">
    </a>
    <a href="https://github.com/tianyi0216/EvoPool">
      <img src="https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github" alt="Code">
    </a>
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
    <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch">
  </p>

</div>

---

## Overview

EvoPool grows a pool of programmatic annotators by iteratively prompting an LLM, then aggregates their votes into training labels for a downstream model. A 4-agent loop (Generator -> Improver -> Refiner -> selection gate) evolves the pool across iterations, and an embedding-aware aggregator converts noisy votes into clean labels that match or beat direct LLM annotation at a fraction of the cost.

## Setup

```bash
conda create -n evopool python=3.11 -y
conda activate evopool
# GPU users: install a CUDA-matched torch wheel from https://pytorch.org first.
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...   # required for the Generator / Improver agents
```


## Quick start

```bash
# 1. Convert raw downloads into processed JSONL splits.
bash process_data.sh

# 2. Run the 4-agent annotator-evolution pipeline + aggregator.
bash run_evopool.sh

# 3. Train the downstream classifier on the aggregated labels.
bash train_downstream.sh
```

Each launcher accepts an optional config path:

```bash
bash run_evopool.sh configs/fever_roberta.yaml
```

## Installation notes

- A CUDA-enabled `torch` build is required for downstream training (CFT / LoRA). Pick the wheel for your CUDA version from [pytorch.org](https://pytorch.org) before running `pip install -r requirements.txt`.
- `sentence-transformers` will download the `all-MiniLM-L6-v2` embedder on first use (~80 MB).
- Reasoning models (o3, gpt-5) are auto-detected from `OPENAI_API_KEY` and dispatched through the reasoning API.

## Configuration

All runtime behavior is driven by a single YAML file (`config.yaml` by default). Top-level groups:

| Key | Purpose |
|-----|---------|
| `dataset`, `data_dir`, `n_classes`, `multi_label`, `task_family` | Which task to run, resolves prompt + evaluator automatically |
| `pipeline.{generator,improver,refiner,selection_gate,query_selection}` | 4-agent loop hyperparameters |
| `pipeline.memory_level` | `0` = Stateless agents `>=2` enables Reflector / shared memory |
| `aggregator` | `evoagg` (paper default) or `mv` (majority-vote baseline) |
| `evoagg.{embedder,cv_folds}` | EvoAgg settings |
| `downstream.model` | `roberta-large` (full fine-tune), `qwen3-1.7b`, or `llama-3.1-8b` (LoRA) |

To switch datasets, copy `config.yaml` to `configs/my_run.yaml`, change `dataset` / `data_dir` / `n_classes` / `multi_label` / `task_family`, and pass it to the launchers. Pre-built configs for ChemProt / FEVER / PubMed across RoBERTa / Qwen / Llama backbones live in `configs/`.

## Datasets

| Dataset | Where to download | Notes |
|---------|-------------------|-------|
| **ChemProt** | WRENCH benchmark: <https://github.com/JieyuZ2/wrench> (`datasets/chemprot/`) | Drop the `train.json` / `valid.json` / `test.json` into `data/raw/chemprot/`. 10-class biomedical relation extraction. |
| **FEVER** | Hugging Face: `copenlu/fever_gold_evidence` (loads via `datasets.load_dataset`) | No manual download needed. Stratified subsample to 10k/1k/7.6k. Set `fever_enriched: true` to add the 5 metadata features (A/B/C/D/E). |
| **PubMed (14-class multi-label)** | Kaggle: <https://www.kaggle.com/datasets/owaiskhan9654/pubmed-multilabel-text-classification> | Download the zip and point `pubmed_kaggle_zip` at it. |

After raw inputs are in place, `bash process_data.sh` writes processed splits to `data/processed/<dataset>/{train,val,test}.jsonl` plus a `dataset.json` manifest.

## Pipeline

1. **Generator** proposes a fresh batch of programmatic annotators each iteration (lexical prompt for classification tasks, verification prompt for FEVER-family).
2. **Improver** allocates a per-class call budget to attack the lowest-F1 classes; widens coverage on long-tail labels.
3. **Refiner** consolidates / deduplicates the high-precision survivors (active only from iter >= 3 by default).
4. **Selection gate** prunes by Jaccard overlap, precision floor, firing floor, and a tiny ablation tolerance (prune-then-diversify via subsumption pruning).
5. **EvoAgg** (the aggregator) fits a 5-fold out-of-fold logistic regression on `[annotator votes | text embedding]` to produce calibrated soft and hard labels (paper default; pass `aggregator: mv` for the majority-vote baseline).
6. **Downstream** trains RoBERTa-large (full FT, with an optional clean-fine-tune curve) or a Qwen / Llama LoRA adapter on the aggregated labels.

## Reproducing paper results

Defaults in `config.yaml` already match the headline cell:

- Pipeline = Stateless Agent (memory_level = 0, Refiner activates at iter >= 3, subsumption pruning on)
- Generator backbone = `gpt-4o-mini`, temperature = 0.5
- 12 iterations, seed = 42
- Aggregator = EvoAgg with 5-fold OOF
- Downstream = RoBERTa-large full fine-tune with the CFT curve `{25, 50, 100, 200, 500, 1000}`

Run end-to-end with the three launchers above.

## Citation

```bibtex
@misc{xu2026evopoolevolutionaryprogrammaticannotation,
      title={EvoPool: Evolutionary Programmatic Annotation for Label-Efficient Specialized Supervision}, 
      author={Tianyi Xu and Yaolun Zhang and Xuan Ouyang and Huazheng Wang},
      year={2026},
      eprint={2606.01617},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.01617}, 
}
```

## License

MIT. See [LICENSE](LICENSE).
