"""EvoPool: 4-agent LF-evolution loop (Generator + Improver + Refiner + optional Reflector).

Single unified pipeline that auto-dispatches between single-label and multi-label
task types based on src.tasks.configs.get_task_config(task).task_type.

Production defaults = D_v6 / L0 (Darwinian; no memory):
  - gpt-4o-mini, temperature=0.5, n_iters=12, seed=42
  - Generator: 18 calls/iter, min_prec=0.30, min_fires=5
  - Improver:  6 calls/iter, min_prec=0.25, min_fires=5 (C1_v2 per-class budget)
  - Refiner:   min_prec=0.55, ref_min_iter=3, max_jaccard=0.95
  - Selection: jaccard=0.95, prec=0.25, fires=5, ablation_drop_threshold=0.001
  - C3 pool prune ON, memory_level=0 (Reflector OFF)
"""
