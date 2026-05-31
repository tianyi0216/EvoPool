"""EvoPool: aggregator package — combines per-annotator votes into final pseudo-labels.

Two production paths:
  - majority_vote: simple per-row vote count (MV / WV passthrough)
  - evoagg:        learned text-aware aggregator (sentence-BERT embeddings + one-hot
                   votes -> LogisticRegressionCV with 5-fold OOF correction)

The active aggregator is selected via `config.aggregator.method` in the project
config.yaml. The entry-point CLI is `src.aggregator.run_aggregator`.
"""
from __future__ import annotations

from .evoagg import (
    fit_predict_evoagg,
    fit_predict_evoagg_multilabel,
    dump_evoagg_labels,
)
from .majority_vote import dump_majority_vote_labels

__all__ = [
    "fit_predict_evoagg",
    "fit_predict_evoagg_multilabel",
    "dump_evoagg_labels",
    "dump_majority_vote_labels",
]
