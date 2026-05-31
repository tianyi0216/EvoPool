"""EvoPool: Claim-verification Generator prompt for NLI-style tasks.

Used by the pipeline when ``task_family == 'verification'`` (e.g. FEVER).

Difference from ``lexical.py``: claim verification is a CLAIM-vs-EVIDENCE
COMPARISON, not a single-text lexical pattern match. The pre-computed
metadata features (token_overlap, entity_containment, negation_mismatch,
nli_contradict, etc.) ARE the right signal — text regex on the combined
"Claim: ... Evidence: ..." string mostly underperforms metadata composition.

We push the LLM toward:
- Compositional helpers OVER METADATA, not over raw text.
- Class 1 (Refutes) is the bottleneck — give it multiple complementary
  signals (negation flip, antonym, NLI contradict, numeric mismatch, all
  with topic-relevance gate).
- Class 2 (NotEnoughInfo) needs LOW topical overlap + no clear contradiction.
- Class 0 (Supports) needs HIGH overlap + absence of contradiction.
"""

VERIFICATION_INSTRUCTION = """\
WRITE COMPOSITIONAL ANNOTATORS OVER PRE-COMPUTED METADATA FEATURES.

PHILOSOPHY: claim verification is a COMPARISON between claim and evidence.
The metadata features above (token_overlap, entity_containment, negation_mismatch,
antonym_present, numbers_mismatch, and — if present — nli_contradict, nli_entail,
sim_semantic, refutes_phrase_count, etc.) are PRE-COMPUTED COMPARISONS. They are
the right primitive. Raw regex on the combined "Claim: ... Evidence: ..." string
usually underperforms because the same word appearing in both sides means nothing
without comparison.

REQUIREMENTS
- Define helper boolean predicates as module-level functions over METADATA, e.g.
  ``has_strong_negation_flip(meta)``, ``has_topic_drift(meta)``,
  ``has_numeric_disagreement(meta)``, ``has_high_overlap_no_contradiction(meta)``.
- Each helper should compose 2-3 metadata signals (not just one).
- For OPTIONAL enriched features (nli_contradict, nli_entail, sim_semantic,
  refutes_phrase_count, supports_phrase_count, nei_phrase_count, antonym_score_ext,
  claim_subj/verb/obj, *_in_evidence): use ``meta.get(key, default)`` so annotators
  stay robust when those features are absent.
- Each ``lf_*`` must call AT LEAST 2 helper predicates AND combine with at least
  one direct metadata threshold check.
- ``return`` integer class id (0=Supports, 1=Refutes, 2=NotEnoughInfo); ``return ABSTAIN`` otherwise.

CLASS-1 (REFUTES) IS THE BOTTLENECK — write MORE Refutes annotators than the other classes
(2-3x as many). For each Refutes annotator, combine:
  (a) topic-relevance signal (entity_containment > 0.3 OR token_overlap > 0.3) —
      Refutes requires the evidence to be ON-TOPIC, not off-topic.
  (b) contradiction signal (>=1 of):
      - negation_mismatch=True
      - numbers_mismatch=True
      - antonym_present=True
      - meta.get("nli_contradict", 0) > 0.5            # strongest single signal
      - meta.get("refutes_phrase_count", 0) >= 1
      - meta.get("antonym_score_ext", 0) >= 1

EXAMPLE STRUCTURE

    def has_strong_negation_flip(meta):
        # negation appears on ONE side AND topic is shared
        return bool(meta.get("negation_mismatch")) and \\
               meta.get("entity_containment", 0.0) > 0.3

    def has_nli_contradict_signal(meta):
        return meta.get("nli_contradict", 0.0) > 0.5

    def has_numeric_disagreement(meta):
        return bool(meta.get("numbers_mismatch")) and \\
               meta.get("has_numbers_both", False)

    def has_antonym_on_topic(meta):
        return (meta.get("antonym_present", False) or
                meta.get("antonym_score_ext", 0) >= 1) and \\
               meta.get("entity_overlap", 0.0) > 0.2

    def has_high_overlap_clean(meta):
        # high topic overlap AND no contradiction signal
        return (meta.get("claim_in_evidence", 0.0) > 0.5 and
                not meta.get("negation_mismatch", False) and
                not meta.get("numbers_mismatch", False) and
                not meta.get("antonym_present", False))

    def has_nli_entail_signal(meta):
        return meta.get("nli_entail", 0.0) > 0.6

    def has_topic_drift(meta):
        return (meta.get("token_overlap", 0.0) < 0.18 and
                meta.get("entity_containment", 0.0) < 0.2 and
                meta.get("num_shared_entities", 0) == 0)

    # -- REFUTES (write 2-3 distinct annotators) --
    def lf_refutes_negation_flip(ex):
        meta = ex.get("metadata", {}) or {}
        if has_strong_negation_flip(meta) and not has_high_overlap_clean(meta):
            return 1
        return ABSTAIN

    def lf_refutes_nli_contradict(ex):
        meta = ex.get("metadata", {}) or {}
        if has_nli_contradict_signal(meta) and meta.get("entity_containment", 0.0) > 0.3:
            return 1
        return ABSTAIN

    def lf_refutes_numeric_or_antonym(ex):
        meta = ex.get("metadata", {}) or {}
        if (has_numeric_disagreement(meta) or has_antonym_on_topic(meta)) and \\
           meta.get("token_overlap", 0.0) > 0.25:
            return 1
        return ABSTAIN

    # -- SUPPORTS --
    def lf_supports_high_overlap_no_neg(ex):
        meta = ex.get("metadata", {}) or {}
        if has_high_overlap_clean(meta) and meta.get("entity_containment", 0.0) > 0.4:
            return 0
        return ABSTAIN

    def lf_supports_nli_entail(ex):
        meta = ex.get("metadata", {}) or {}
        if has_nli_entail_signal(meta) and not has_strong_negation_flip(meta):
            return 0
        return ABSTAIN

    # -- NEI --
    def lf_nei_topic_drift(ex):
        meta = ex.get("metadata", {}) or {}
        if has_topic_drift(meta) and not has_strong_negation_flip(meta):
            return 2
        return ABSTAIN

GUIDANCE
- Generate 7-10 lf_* functions; lean MORE on Refutes (2-3 distinct annotators).
- Vary the combinations - don't write 3 annotators that all check
  (neg_mismatch + entity_containment); mix in nli_contradict, antonym_score_ext,
  refutes_phrase_count for diversity.
- It is OK if a helper is shared across annotators.
- DO NOT just regex the raw ex["text"]; the metadata IS your primitive.

Output everything in ONE python ```code block```.
"""
