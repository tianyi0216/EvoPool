"""EvoPool: pipeline runner. Reads config.yaml and invokes the orchestrator
with translated CLI arguments.

Usage:
    python -m src.pipeline.run --config config.yaml

The dispatcher exists so the legacy ~50-flag orchestrator stays untouched while
end users only need a single YAML.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.utils.config_loader import get, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cli_args = _build_orchestrator_args(cfg)

    sys.argv = ["orchestrator"] + cli_args
    from src.pipeline.orchestrator import main as orchestrator_main

    orchestrator_main()


def _build_orchestrator_args(cfg: Dict[str, Any]) -> List[str]:
    """Translate config.yaml -> orchestrator.py CLI args."""
    cli: List[str] = []

    # Task / output / data
    task = get(cfg, "dataset.name")
    if task:
        cli += ["--task", str(task)]
    out_root = get(cfg, "pipeline.out_root") or get(cfg, "output.pipeline_dir") or get(cfg, "output.base_dir")
    if out_root is None:
        raise SystemExit("config.pipeline.out_root (or output.base_dir) must be set.")
    cli += ["--out_root", str(out_root)]

    # Core hyperparams
    _add_if_set(cli, "--n_iters", get(cfg, "pipeline.n_iters"))
    _add_if_set(cli, "--model", get(cfg, "pipeline.model"))
    _add_if_set(cli, "--temperature", get(cfg, "pipeline.temperature"))
    _add_if_set(cli, "--seed", get(cfg, "pipeline.seed"))
    _add_if_set(cli, "--memory_level", get(cfg, "pipeline.memory_level"))
    _add_if_set(cli, "--n_classes", get(cfg, "dataset.n_classes"))

    # Generator
    _add_if_set(cli, "--gen_n_calls", get(cfg, "pipeline.generator.n_calls"))
    _add_if_set(cli, "--gen_min_precision", get(cfg, "pipeline.generator.min_precision"))
    _add_if_set(cli, "--gen_min_fires", get(cfg, "pipeline.generator.min_fires"))

    # Improver
    _add_if_set(cli, "--imp_n_calls", get(cfg, "pipeline.improver.n_calls"))
    _add_if_set(cli, "--imp_min_precision", get(cfg, "pipeline.improver.min_precision"))
    _add_if_set(cli, "--imp_min_fires", get(cfg, "pipeline.improver.min_fires"))
    _add_if_set(cli, "--imp_max_jaccard_overlap", get(cfg, "pipeline.improver.max_jaccard_overlap"))
    _add_if_set(cli, "--imp_max_train_val_prec_gap", get(cfg, "pipeline.improver.max_train_val_prec_gap"))

    # C1_v2 per-class budget Improver (D_v6 production default = True)
    if get(cfg, "pipeline.improver.per_class_budget", True):
        cli += ["--c1v2_per_class_improver"]
    else:
        cli += ["--disable_c1v2"]
    _add_if_set(cli, "--c1v2_f1_threshold", get(cfg, "pipeline.improver.f1_threshold"))
    _add_if_set(cli, "--c1v2_max_targets", get(cfg, "pipeline.improver.max_targets"))
    _add_if_set(cli, "--c1v2_n_pos_examples", get(cfg, "pipeline.improver.n_pos_examples"))

    # Refiner
    _add_if_set(cli, "--ref_filter_min_precision", get(cfg, "pipeline.refiner.filter_min_precision"))
    _add_if_set(cli, "--ref_max_jaccard_overlap", get(cfg, "pipeline.refiner.max_jaccard_overlap"))
    _add_if_set(cli, "--ref_min_iter", get(cfg, "pipeline.refiner.min_iter"))

    # Ablation
    if get(cfg, "pipeline.ablation.enable_post_iter", True):
        cli += ["--enable_post_iter_ablation"]
    else:
        cli += ["--disable_post_iter_ablation"]
    _add_if_set(cli, "--ablation_drop_threshold", get(cfg, "pipeline.ablation.drop_threshold"))
    _add_if_set(cli, "--min_iter_gain", get(cfg, "pipeline.ablation.min_iter_gain"))

    # Pruning (C3 + dropout)
    if get(cfg, "pipeline.pruning.c3_pool_pruning", True):
        cli += ["--c3_pool_pruning"]
    else:
        cli += ["--disable_c3_pool_pruning"]
    _add_if_set(cli, "--class_dropout_after", get(cfg, "pipeline.pruning.class_dropout_after"))
    _add_if_set(cli, "--do_no_harm_tolerance", get(cfg, "pipeline.pruning.do_no_harm_tolerance"))

    # Query selection
    _add_if_set(cli, "--query_selection_method", get(cfg, "pipeline.query_selection.method"))
    _add_if_set(cli, "--query_k_max", get(cfg, "pipeline.query_selection.batch_size"))

    # Cache
    _add_if_set(cli, "--cache_root", get(cfg, "pipeline.cache_root"))

    return cli


def _add_if_set(cli: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cli += [flag, str(value)]


if __name__ == "__main__":
    main()
