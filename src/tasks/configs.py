"""EvoPool: task configurations.

Defines dataset settings (label names, description, metadata features, prompt
guardrails) for each released task. Components import via ``get_task_config``.

Two derived fields auto-set by task name:
  - ``multi_label`` (bool): True for tasks with ``true_labels: List[int]``.
  - ``task_family`` (str): ``classification`` or ``verification``; drives
    which Generator prompt template is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


_VERIFICATION_TASKS = {"fever"}
_MULTI_LABEL_TASKS = {"pubmed"}


@dataclass
class TaskConfig:
    """Configuration for a classification task (single-label or multi-label)."""
    task_name: str
    task_description: str          # short description for LLM prompts
    label_names: Dict[int, str]    # {0: "ham", 1: "spam"} etc.
    num_classes: int
    valid_labels: Set[int]         # set of valid label ints
    bool_features: List[str]       # metadata boolean features for segmentation
    # per-class guardrail hints for LLM prompt (optional)
    class_guardrails: Dict[int, str] = field(default_factory=dict)
    # metadata field descriptions shown in LLM prompt
    metadata_fields_description: str = ""
    multi_label: bool = False
    task_family: str = "classification"   # "classification" | "verification"

    def __post_init__(self) -> None:
        # Auto-fill multi_label / task_family by task name so callers that
        # construct a TaskConfig without setting these still get sensible defaults.
        if not self.multi_label and self.task_name in _MULTI_LABEL_TASKS:
            self.multi_label = True
        if self.task_family == "classification" and self.task_name in _VERIFICATION_TASKS:
            self.task_family = "verification"

    @property
    def label_names_str(self) -> str:
        """Format label schema for prompts, e.g. 'ham = 0, spam = 1'"""
        return ", ".join(f"{name} = {label}" for label, name in sorted(self.label_names.items()))


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "chemprot": TaskConfig(
        task_name="chemprot",
        task_description=(
            "ChemProt chemical-protein relation extraction (10-class biomedical).\n"
            "Given a sentence describing a chemical-protein interaction, classify the relation type.\n"
            "Each example contains two highlighted entities (a chemical and a protein) and the sentence.\n"
            "The annotator should analyze the sentence to determine the relation between the entities."
        ),
        label_names={
            0: "Part-of",
            1: "Regulator",
            2: "Upregulator",
            3: "Downregulator",
            4: "Agonist",
            5: "Antagonist",
            6: "Modulator",
            7: "Cofactor",
            8: "Substrate/Product",
            9: "NOT",
        },
        num_classes=10,
        valid_labels=set(range(10)),
        bool_features=[],
        class_guardrails={
            0: (
                "Part-of: The chemical is a structural component of the protein.\n"
                "Keywords: 'amino acid', 'residue', 'domain', 'subunit', 'moiety', 'replace', 'mutant',\n"
                "  'mutation', 'substitution', 'which is present in', 'constituent', 'component'\n"
                "Pattern: entity1 'is part of' entity2, 'residue in', 'domain of'\n"
                "NOTE: Very rare class. Be precise — don't confuse with general binding."
            ),
            1: (
                "Regulator: The chemical regulates the protein (general regulation, not up/down specific).\n"
                "Keywords: 'bind', 'interact', 'affinity', 'target', 'receptor', 'ligand', 'associate',\n"
                "  'complex', 'recognize', 'selective for', 'specific to'\n"
                "Pattern: entity1 'binds to' entity2, 'interaction between', 'affinity for'\n"
                "NOTE: Largest class. Use binding/interaction language. Exclude if up/down/agonist is clear."
            ),
            2: (
                "Upregulator: The chemical increases/activates the protein activity.\n"
                "Keywords: 'activate', 'increase', 'enhance', 'induce', 'stimulate', 'upregulate',\n"
                "  'up-regulate', 'promote', 'elevate', 'potentiate', 'augment'\n"
                "Pattern: entity1 'activates' entity2, 'increased expression of', 'induced by'\n"
                "NOTE: Distinguish from Agonist (receptor-specific activation)."
            ),
            3: (
                "Downregulator: The chemical decreases/inhibits the protein activity.\n"
                "Keywords: 'inhibit', 'reduce', 'decrease', 'suppress', 'downregulate', 'down-regulate',\n"
                "  'block', 'attenuate', 'abolish', 'prevent', 'impair', 'diminish'\n"
                "Pattern: entity1 'inhibits' entity2, 'reduced activity of', 'blocked by'\n"
                "NOTE: Second largest class. 'inhibit' is the strongest signal. Distinguish from Antagonist."
            ),
            4: (
                "Agonist: The chemical is an agonist of the protein/receptor.\n"
                "Keywords: 'agonist', 'agonism', 'agonistic', 'partial agonist', 'full agonist'\n"
                "Pattern: entity1 'is an agonist of' entity2, 'agonist at the' entity2 'receptor'\n"
                "NOTE: Rare class. The word 'agonist' is almost always present. Very specific."
            ),
            5: (
                "Antagonist: The chemical is an antagonist of the protein/receptor.\n"
                "Keywords: 'antagonist', 'antagonism', 'antagonistic', 'blocker', 'competitive antagonist'\n"
                "Pattern: entity1 'is an antagonist of' entity2, 'antagonist at'\n"
                "NOTE: The word 'antagonist' is the strongest signal. Don't confuse with general inhibition."
            ),
            6: (
                "Modulator: The chemical modulates the protein activity (allosteric or general modulation).\n"
                "Keywords: 'modulate', 'modulator', 'allosteric', 'potentiate', 'positive allosteric modulator',\n"
                "  'negative modulator', 'non-competitive'\n"
                "Pattern: 'allosteric modulator of', 'modulates the activity of'\n"
                "NOTE: Very rare class (N~55). 'allosteric' and 'modulator' are key signals."
            ),
            7: (
                "Cofactor: The chemical is a cofactor of the protein/enzyme.\n"
                "Keywords: 'cofactor', 'coenzyme', 'prosthetic group', 'essential for catalysis'\n"
                "Pattern: entity1 'is a cofactor of' entity2, 'cofactor for'\n"
                "NOTE: Extremely rare class (N~51). The word 'cofactor' is almost always present."
            ),
            8: (
                "Substrate/Product: The chemical is a substrate or product of the protein/enzyme.\n"
                "Keywords: 'substrate', 'product', 'catalyze', 'catalyse', 'metabolize', 'convert',\n"
                "  'enzyme', 'transport', 'transporter', 'carrier', 'metabolism', 'biotransformation'\n"
                "Pattern: 'substrate of', 'catalyzed by', 'metabolized by', 'transported by'\n"
                "NOTE: 'substrate', 'catalyz/catalys', 'metaboliz', 'transport' are strong signals."
            ),
            9: (
                "NOT: No specific chemical-protein relation, or the relation doesn't fit other categories.\n"
                "Keywords: 'not', 'no effect', 'unrelated', 'independent', 'no significant',\n"
                "  'did not affect', 'no interaction', 'failed to'\n"
                "Pattern: 'not' near entities, negation between entity mentions\n"
                "NOTE: Use with caution. Only label as NOT when there is clear negation or no relation."
            ),
        },
        metadata_fields_description=(
            '  ex["text"]: biomedical sentence describing chemical-protein interaction\n'
            '  ex["metadata"]["text_normalized"]: lowercased normalized text\n'
            '  ex["metadata"]["entity1"]: the chemical entity name\n'
            '  ex["metadata"]["entity2"]: the protein entity name\n'
            '  ex["metadata"]["span1"]: [start, end] character offsets of entity1\n'
            '  ex["metadata"]["span2"]: [start, end] character offsets of entity2\n'
            '  ex["metadata"]["num_chars"], num_words: text length stats'
        ),
    ),
    "fever": TaskConfig(
        task_name="fever",
        task_description=(
            "FEVER fact verification (3-class: Supports, Refutes, NotEnoughInfo).\n"
            "Given a claim and evidence from Wikipedia, determine if the evidence supports,\n"
            "refutes, or is not enough to verify the claim.\n"
            "Each example contains a claim and one or more evidence sentences."
        ),
        label_names={0: "Supports", 1: "Refutes", 2: "NotEnoughInfo"},
        num_classes=3,
        valid_labels={0, 1, 2},
        bool_features=[
            "negation_in_claim", "negation_in_evidence", "negation_mismatch",
            "has_numbers_claim", "has_numbers_evidence", "has_numbers_both",
            "numbers_match", "numbers_mismatch", "antonym_present",
        ],
        class_guardrails={
            0: (
                "Supports: The evidence CONFIRMS the claim as true.\n"
                "USE PRE-COMPUTED VERIFICATION FEATURES (not raw regex on text!):\n"
                "  1. High token_overlap (>0.4) + high claim_in_evidence (>0.5) → claim's words appear in evidence\n"
                "  2. High entity_containment (>0.5) → claim's entities found in evidence\n"
                "  3. numbers_match=True → numeric values agree\n"
                "  4. negation_mismatch=False → no contradicting negation\n"
                "  5. antonym_present=False → no antonym contradiction\n"
                "COMBINE multiple features for higher precision, e.g.:\n"
                "  if meta['claim_in_evidence'] > 0.6 and not meta['negation_mismatch'] and not meta['numbers_mismatch']: return 0\n"
                "IMPORTANT: Use meta = ex.get('metadata', {}). Features are pre-computed floats/bools."
            ),
            1: (
                "Refutes: The evidence CONTRADICTS the claim.\n"
                "USE PRE-COMPUTED VERIFICATION FEATURES:\n"
                "  1. negation_mismatch=True → one side has negation, other doesn't\n"
                "  2. numbers_mismatch=True → both have numbers but they differ\n"
                "  3. antonym_present=True → claim and evidence contain antonym pair\n"
                "  4. High entity_overlap but negation_mismatch → same topic, opposite conclusion\n"
                "  5. Combine: entity_containment > 0.3 AND (negation_mismatch OR numbers_mismatch)\n"
                "IMPORTANT: Refutation needs BOTH topic relevance AND contradiction signal.\n"
                "  if meta['entity_containment'] > 0.3 and meta['negation_mismatch']: return 1"
            ),
            2: (
                "NotEnoughInfo: The evidence is INSUFFICIENT to verify or refute.\n"
                "USE PRE-COMPUTED VERIFICATION FEATURES:\n"
                "  1. Low token_overlap (<0.2) → claim and evidence discuss different things\n"
                "  2. Low entity_containment (<0.2) → claim's entities not in evidence\n"
                "  3. num_shared_entities == 0 → no entity overlap at all\n"
                "  4. Low claim_in_evidence (<0.3) AND no negation_mismatch → unrelated\n"
                "  5. Combine: token_overlap < 0.15 and entity_overlap < 0.1\n"
                "IMPORTANT: NEI = evidence is about a DIFFERENT topic or missing key details.\n"
                "  if meta['token_overlap'] < 0.15 and meta['entity_overlap'] < 0.1: return 2"
            ),
        },
        metadata_fields_description=(
            '  ex["text"]: "Claim: {claim}\\nEvidence: {evidence}" (combined)\n'
            '  ex["metadata"]["claim"]: the claim to be verified\n'
            '  ex["metadata"]["evidence"]: Wikipedia evidence sentence(s)\n'
            '  ## PRE-COMPUTED VERIFICATION FEATURES (use these!):\n'
            '  ex["metadata"]["token_overlap"]: float, Jaccard of content words (0-1)\n'
            '  ex["metadata"]["claim_in_evidence"]: float, fraction of claim words found in evidence (0-1)\n'
            '  ex["metadata"]["evidence_in_claim"]: float, fraction of evidence words found in claim (0-1)\n'
            '  ex["metadata"]["entity_overlap"]: float, Jaccard of entity mentions (0-1)\n'
            '  ex["metadata"]["entity_containment"]: float, fraction of claim entities in evidence (0-1)\n'
            '  ex["metadata"]["num_shared_entities"]: int, count of shared entities\n'
            '  ex["metadata"]["num_claim_entities"]: int, count of entities in claim\n'
            '  ex["metadata"]["claim_entities"]: list of str, entity mentions in claim\n'
            '  ex["metadata"]["evidence_entities"]: list of str, entity mentions in evidence\n'
            '  ex["metadata"]["has_numbers_claim"]: bool\n'
            '  ex["metadata"]["has_numbers_evidence"]: bool\n'
            '  ex["metadata"]["has_numbers_both"]: bool, both sides have numbers\n'
            '  ex["metadata"]["numbers_match"]: bool, at least one number matches\n'
            '  ex["metadata"]["numbers_mismatch"]: bool, both have numbers but none match\n'
            '  ex["metadata"]["negation_in_claim"]: bool\n'
            '  ex["metadata"]["negation_in_evidence"]: bool\n'
            '  ex["metadata"]["negation_mismatch"]: bool, one side has negation, other does not\n'
            '  ex["metadata"]["antonym_present"]: bool, known antonym pair detected\n'
            '  ex["metadata"]["claim_word_count"]: int\n'
            '  ex["metadata"]["evidence_word_count"]: int\n'
            '  ex["metadata"]["length_ratio"]: float, evidence_words / claim_words\n'
            '  ex["metadata"]["text_normalized"]: normalized combined text\n'
            '  ex["metadata"]["has_url"], has_email\n'
            '  ## OPTIONAL ENRICHED FEATURES (present only if dataset enriched by\n'
            '     scripts/enrich_fever_metadata.py; use via .get(key, default) so annotators\n'
            '     stay robust on base data):\n'
            '  ex["metadata"].get("sim_semantic", 0.0): float, sentence-transformer cosine of claim vs. evidence (0-1)\n'
            '    -> High (>0.7) + not negation_mismatch => Supports\n'
            '    -> Low  (<0.3) => NotEnoughInfo (topically unrelated)\n'
            '  ex["metadata"].get("sim_semantic_bin", ""): str in {very_low,low,med,high}\n'
            '  ex["metadata"].get("nli_entail", 0.0): float, pretrained NLI entailment prob (0-1)\n'
            '    -> >0.6 => Supports (evidence entails claim)\n'
            '  ex["metadata"].get("nli_contradict", 0.0): float, pretrained NLI contradiction prob (0-1)\n'
            '    -> >0.5 => Refutes  (evidence contradicts claim) -- single strongest feature\n'
            '  ex["metadata"].get("nli_neutral", 0.0): float, pretrained NLI neutral prob (0-1)\n'
            '    -> >0.6 => NotEnoughInfo\n'
            '  ex["metadata"].get("claim_subj","")/.get("claim_verb","")/.get("claim_obj",""): str, spaCy-parsed SVO from claim\n'
            '  ex["metadata"].get("subj_in_evidence",False)/.get("verb_in_evidence",False)/.get("obj_in_evidence",False): bool\n'
            '  ex["metadata"].get("antonym_score_ext", 0): int, WordNet antonym pair count between claim/evidence\n'
            '    -> >=1 => Refutes (explicit antonym pair found)\n'
            '  ex["metadata"].get("refutes_phrase_count", 0): int, hits on extended 40+ refutation-cue phrase bank\n'
            '  ex["metadata"].get("supports_phrase_count", 0): int, hits on supports-cue phrase bank\n'
            '  ex["metadata"].get("nei_phrase_count", 0): int, hits on hedge/uncertainty phrase bank\n'
            '    -> prefer COMBINING these with existing features, e.g.\n'
            '       if meta.get("nli_contradict",0)>0.5 and meta.get("entity_containment",0)>0.3: return 1'
        ),
    ),

    "pubmed": TaskConfig(
        task_name="pubmed",
        task_description=(
            "PubMed multi-label MeSH top-level classification (14 categories: A-N, no K/V).\n"
            "Given a PubMed abstract, identify ALL relevant MeSH top categories.\n"
            "Each doc covers MULTIPLE categories (avg 5.72/14 = dense multi-label).\n"
            "Distribution dominated by B (Organisms, 93%), E (Techniques, 79%),\n"
            "D (Chemicals, 62%); smaller categories (L, Z) sparse. Write annotators specific\n"
            "to one MeSH category at a time; aggregator OR-combines votes."
        ),
        label_names={
            0:  "A_Anatomy",
            1:  "B_Organisms",
            2:  "C_Diseases",
            3:  "D_Chemicals_and_Drugs",
            4:  "E_Analytical_Diagnostic_Therapeutic_Techniques",
            5:  "F_Psychiatry_and_Psychology",
            6:  "G_Phenomena_and_Processes",
            7:  "H_Disciplines_and_Occupations",
            8:  "I_Anthropology_Education_Sociology_Social_Phenomena",
            9:  "J_Technology_Industry_Agriculture",
            10: "L_Information_Science",
            11: "M_Named_Groups",
            12: "N_Health_Care",
            13: "Z_Geographicals",
        },
        num_classes=14,
        valid_labels=set(range(14)),
        bool_features=[],
        class_guardrails={
            0:  "A Anatomy: body parts/structures/tissues. Keywords: 'tissue','organ','cell','muscle','bone','nerve','vessel','epithelium','membrane'.",
            1:  "B Organisms (93% of papers!): any living entity. Keywords: 'human','mouse','rat','bacteria','virus','species','animal','patient','subject'.",
            2:  "C Diseases. Keywords: 'disease','disorder','syndrome','infection','cancer','diabetes','injury','pathology'.",
            3:  "D Chemicals and Drugs. Keywords: 'drug','protein','enzyme','receptor','peptide','antibody','compound','mg/kg','dose'.",
            4:  "E Analytical/Diagnostic/Therapeutic Techniques. Keywords: 'PCR','MRI','CT','ELISA','assay','surgery','therapy','treatment','procedure'.",
            5:  "F Psychiatry/Psychology. Keywords: 'depression','anxiety','psychiatric','cognitive','behavior','psychological','schizophrenia'.",
            6:  "G Phenomena and Processes (bio). Keywords: 'metabolism','transcription','signaling','apoptosis','expression','pathway','regulation'.",
            7:  "H Disciplines and Occupations. Keywords: 'physician','specialty','professional','training','clinic'.",
            8:  "I Anthropology/Sociology. Keywords: 'social','cultural','ethnic','race','community','demographic','population','public'.",
            9:  "J Technology/Industry/Agriculture. Keywords: 'industrial','manufacturing','agriculture','food processing','occupational'.",
            10: "L Information Science. Keywords: 'database','bioinformatics','data analysis','algorithm','model','computational'.",
            11: "M Named Groups (cohorts). Keywords: 'children','elderly','women','adolescents','infants','pregnant','cohort'.",
            12: "N Health Care delivery. Keywords: 'healthcare','health policy','insurance','quality of life','cost','utilization'.",
            13: "Z Geographicals. Keywords: 'America','Europe','Asia','rural','urban','region','country','state','community-based'.",
        },
        metadata_fields_description=(
            '  ex["text"]: PubMed abstract\n'
            '  ex["true_labels"]: List[int] subset of 14 MeSH top categories\n'
            '  ex["true_label"]: int — first of true_labels (single-label projection)\n'
        ),
    ),
}


def get_task_config(task_name: str) -> TaskConfig:
    """Get task config by name. Raises KeyError if not found."""
    if task_name not in TASK_CONFIGS:
        available = ", ".join(sorted(TASK_CONFIGS.keys()))
        raise KeyError(f"Unknown task: {task_name!r}. Available: {available}")
    return TASK_CONFIGS[task_name]
