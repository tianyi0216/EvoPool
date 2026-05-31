"""EvoPool: 4-agent annotator-evolution loop (Generator + Improver + Refiner + optional Reflector).

Single unified pipeline that auto-dispatches between single-label and multi-label
task types based on src.tasks.configs.get_task_config(task).task_type.

Production defaults: gpt-4o-mini, T=0.5, n_iters=12, seed=42. See config.yaml
for full hyperparameters.
"""
