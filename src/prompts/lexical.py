"""EvoPool: Lexical (no-CoT) Generator prompt for single-text classification tasks.

This is the production prompt for tasks where each example is a single text
span and the annotator decides a class label from lexical/metadata patterns
(e.g. ChemProt relation extraction, AG News topic classification, Banking77
intent classification, DDI drug-drug interaction).

For claim-vs-evidence verification tasks (FEVER, SciFact, VitaminC, ANLI),
use ``verification.py`` instead — its prompt is grounded on pre-computed
comparison features (token_overlap, negation_mismatch, nli_contradict, ...).

The B-axis ablation found this lexical, no-CoT, permissive-helpers prompt
("B1_V1") dominates CoT, rationale, self-debug, distillation and embedding
variants on long-tail biomedical classes. CoT rationalizes wrong predictions
on minority classes; lexical patterns over metadata are the right primitive.
"""

LEXICAL_INSTRUCTION = """\
WRITE COMPOSITIONAL ANNOTATORS WITH MULTI-STEP REASONING.

PHILOSOPHY: each labeling function should be a *compositional Python program*,
not a flat if-else. Use whatever Python feature makes the rule expressive
and correct: helper predicates, regex compilation, for-loops over keyword
banks, list/dict comprehensions, even imports if useful.

REQUIREMENTS
- Define helper boolean predicates as module-level functions (e.g.
  ``has_inhibition_keywords(text)``, ``has_chemical_target(meta)``).
- Define each labeling function as ``lf_<descriptive_name>(ex)``. The
  ``lf_*`` body should COMPOSE the helpers via explicit IF-ELIF-ELSE.
- Each lf_* must call AT LEAST 2 helper predicates AND combine with at
  least one direct check (text pattern or metadata field).
- ``return`` the integer class id when matched, ``return ABSTAIN`` otherwise.
- Use ONLY metadata fields enumerated in metadata_fields_description above
  (do not invent fields like ``entity_overlap`` if they're not listed).

EXAMPLE STRUCTURE (adapt to this task's actual metadata fields):

    import re
    INHIBITION_KW = ['inhibit', 'suppress', 'block', 'reduce', 'attenuate']

    def has_inhibition_keywords(text):
        return any(k in text for k in INHIBITION_KW)

    def has_named_chemical_target(meta):
        e1, e2 = meta.get('entity1', ''), meta.get('entity2', '')
        return bool(e1) and bool(e2) and len(e1) > 1 and len(e2) > 1

    def lf_downregulator_inhibition_compositional(ex):
        meta = ex.get('metadata', {}) or {}
        text = (ex.get('text') or '').lower()
        if has_inhibition_keywords(text) and has_named_chemical_target(meta):
            if 'metabolism' not in text and 'metabolize' not in text:
                return 3
        return ABSTAIN

- Generate 5-8 lf_* functions varying target classes; share helpers freely.
- Imports + helpers + for-loops + regex are all allowed. Be expressive.

Output everything in ONE python ```code block```.
"""
