"""EvoPool: downstream training dispatcher.

Reads config.yaml and dispatches to one of the four concrete trainers:
- src.downstream.train_encoder              (RoBERTa-large full FT, single-label)
- src.downstream.train_encoder_multilabel   (RoBERTa-large full FT, multi-label BCE)
- src.downstream.train_decoder_lora         (Qwen / Llama LoRA, single-label)
- src.downstream.train_decoder_lora_multilabel (Qwen / Llama LoRA, multi-label BCE)

The choice depends on:
- config.downstream.model: roberta | qwen | llama
- task_type resolved from src.tasks.configs (single_label | multi_label)

Usage:
    python -m src.downstream.train --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.tasks.configs import get_task_config
from src.utils.config_loader import get, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    task_name = get(cfg, "dataset.name")
    if not task_name:
        raise SystemExit("config.dataset.name must be set.")
    task_cfg = get_task_config(task_name)
    task_type = getattr(task_cfg, "task_type", "single_label")
    multi_label = task_type == "multi_label" or bool(get(cfg, "dataset.multi_label", False))

    model = (get(cfg, "downstream.model") or "roberta").lower()

    if model == "roberta":
        if multi_label:
            from src.downstream.train_encoder_multilabel import main as trainer_main
            label = "train_encoder_multilabel"
        else:
            from src.downstream.train_encoder import main as trainer_main
            label = "train_encoder"
    elif model in {"qwen", "llama"}:
        if multi_label:
            from src.downstream.train_decoder_lora_multilabel import main as trainer_main
            label = "train_decoder_lora_multilabel"
        else:
            from src.downstream.train_decoder_lora import main as trainer_main
            label = "train_decoder_lora"
    else:
        raise SystemExit(f"Unknown downstream.model: {model!r}. Pick one of: roberta | qwen | llama")

    cli = _build_trainer_args(cfg, model, multi_label)
    print(f"[train] dispatching to {label} with {len(cli) // 2} args", flush=True)

    sys.argv = [label] + cli
    trainer_main()


def _build_trainer_args(cfg: Dict[str, Any], model: str, multi_label: bool) -> List[str]:
    """Translate config.yaml -> the chosen trainer's CLI args.

    Each trainer has a different CLI surface so we branch:
    - train_encoder (roberta + single-label) uses verbose names
      (--num_epochs / --learning_rate / --hard_label_path / --cft).
    - All other trainers (encoder_multilabel + decoder_lora[_multilabel])
      use --ws_epochs / --ws_lr / --grad_accum and read pseudo-labels
      directly from --run_root/iter_*/eval/train_labeled.jsonl.
    """
    cli: List[str] = []
    use_encoder_singlelabel = (model == "roberta" and not multi_label)

    processed_dir = Path(get(cfg, "dataset.processed_dir") or f"data/processed/{get(cfg, 'dataset.name')}")
    cli += ["--train_path", str(processed_dir / "train.jsonl")]
    cli += ["--val_path", str(processed_dir / "val.jsonl")]
    cli += ["--test_path", str(processed_dir / "test.jsonl")]

    out_root = get(cfg, "pipeline.out_root") or get(cfg, "output.pipeline_dir")
    if out_root:
        cli += ["--run_root", str(out_root)]

    out_dir = (
        get(cfg, "downstream.out_dir")
        or get(cfg, "output.downstream_dir")
        or (Path(get(cfg, "output.base_dir", ".")) / "downstream").as_posix()
    )
    cli += ["--out_dir", str(out_dir)]

    model_name = get(cfg, "downstream.model_name")
    if model_name:
        cli += ["--model_name", str(model_name)]

    if multi_label:
        n_classes = get(cfg, "dataset.n_classes")
        if n_classes is not None:
            cli += ["--num_labels", str(n_classes)]

    # Pseudo-label phase hyperparams (encoder vs all-others have different flag names)
    ws = get(cfg, "downstream.ws", {}) or {}
    if use_encoder_singlelabel:
        _add_if_set(cli, "--num_epochs", ws.get("epochs"))
        _add_if_set(cli, "--learning_rate", ws.get("lr"))
        hard_label_path = get(cfg, "downstream.hard_label_path") or get(cfg, "aggregator.dump_dir")
        if hard_label_path:
            cli += ["--hard_label_path", str(hard_label_path)]
    else:
        _add_if_set(cli, "--ws_epochs", ws.get("epochs"))
        _add_if_set(cli, "--ws_lr", ws.get("lr"))

    _add_if_set(cli, "--batch_size", ws.get("batch_size"))
    _add_if_set(cli, "--max_length", ws.get("max_length"))
    _add_if_set(cli, "--seed", get(cfg, "downstream.seed", 42))

    # CFT
    cft = get(cfg, "downstream.cft", {}) or {}
    if cft.get("enabled"):
        if use_encoder_singlelabel:
            cli += ["--cft"]
            _add_if_set(cli, "--cft_epochs", cft.get("epochs"))
            _add_if_set(cli, "--cft_learning_rate", cft.get("lr"))
        else:
            _add_if_set(cli, "--cft_epochs", cft.get("epochs"))
            _add_if_set(cli, "--cft_lr", cft.get("lr"))
            clean_sizes = cft.get("clean_sizes")
            if clean_sizes:
                cli += ["--clean_sizes", ",".join(str(x) for x in clean_sizes)]

    # LoRA (decoder models only)
    if model in {"qwen", "llama"}:
        lora = get(cfg, "downstream.lora", {}) or {}
        _add_if_set(cli, "--lora_r", lora.get("r"))
        _add_if_set(cli, "--lora_alpha", lora.get("alpha"))
        _add_if_set(cli, "--lora_dropout", lora.get("dropout"))
        if "batch_size" in lora:
            cli += ["--batch_size", str(lora["batch_size"])]
        _add_if_set(cli, "--grad_accum", lora.get("grad_accum"))

    return cli


def _add_if_set(cli: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cli += [flag, str(value)]


if __name__ == "__main__":
    main()
