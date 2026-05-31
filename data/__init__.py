"""EvoPool: dataset preparation package.

Submodules:
    prepare_chemprot  - convert WRENCH ChemProt JSON to JSONL
    prepare_fever     - download/convert FEVER (gold evidence) to JSONL
    prepare_pubmed    - parse Kaggle PubMed 14-class multi-label zip to JSONL
    enrich_fever      - optional 5-feature semantic/NLI/syntactic enrichment for FEVER
    prepare           - top-level dispatch (driven by config.yaml)
"""
