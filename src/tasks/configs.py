"""EvoPool: unified task configurations (single-label + multi-label).

Each task config defines dataset-specific settings: label names, task descriptions,
metadata features, and prompt guardrails. All EvoPool components import from
here via get_task_config(task_name).

The two new fields vs the legacy script-side config are:
  - multi_label (bool): True for tasks with true_labels: List[int] (e.g.
    kaggle_pubmed, ohsumed_ml).
  - task_family (str): "classification" | "verification" -- drives which
    Generator prompt template is selected (lexical vs. verification).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


_VERIFICATION_TASKS = {"fever", "vitaminc", "scifact", "anli"}
_MULTI_LABEL_TASKS = {"ohsumed_ml", "kaggle_pubmed"}


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
    "youtube_spam": TaskConfig(
        task_name="youtube_spam",
        task_description="YouTube comment spam detection (binary: ham vs spam).",
        label_names={0: "ham", 1: "spam"},
        num_classes=2,
        valid_labels={0, 1},
        bool_features=[
            "has_url", "has_subscribe", "has_channel",
            "has_check_out", "has_email", "has_like",
        ],
        class_guardrails={
            0: (
                "ham-voting functions MUST abstain on promotional signals "
                "(urls/emails/subscribe/channel/check-out). Include negative guards like:\n"
                '  if meta.get("has_url") or meta.get("has_email") or meta.get("has_subscribe") '
                'or meta.get("has_channel") or meta.get("has_check_out"): return ABSTAIN'
            ),
        },
        metadata_fields_description=(
            '  ex["text"]: raw comment text\n'
            '  ex["metadata"]["text_normalized"]: normalized text\n'
            '  ex["metadata"]["has_url"], has_email, has_subscribe, has_like, has_channel, has_check_out\n'
            '  ex["metadata"]["num_words"], num_exclamation, num_question, num_uppercase'
        ),
    ),

    "imdb": TaskConfig(
        task_name="imdb",
        task_description="IMDB movie review sentiment classification (binary: negative vs positive).",
        label_names={0: "neg", 1: "pos"},
        num_classes=2,
        valid_labels={0, 1},
        bool_features=["has_url", "has_email"],
        class_guardrails={
            0: (
                "Movie reviews are LONG text. AIM FOR DIVERSE STRATEGIES to maximize coverage.\n"
                "Strategy types (use ALL of these, not just keywords):\n"
                "  1. Multi-word phrases: 'waste of time', 'poorly acted', 'complete garbage'\n"
                "  2. Strong sentiment words: 'atrocious', 'unwatchable', 'abysmal', 'horrendous'\n"
                "  3. Negation-aware patterns: negative word NOT preceded by 'not'/'never'\n"
                "  4. Metadata + text combos: e.g., short review (num_words < 80) AND contains 'bad';\n"
                "     many exclamation marks (num_exclamation >= 3) AND contains a negative word\n"
                "  5. Structural patterns: review starts/ends with negative phrase;\n"
                "     ALL-CAPS words combined with negative sentiment\n"
                "  6. Pattern-based: 'poorly (acted|written|directed|executed)',\n"
                "     'waste of (time|money)', 'one of the worst'\n"
                "  7. Rating patterns: '1/10', '1 out of 10', 'zero stars'\n"
                "  8. Combine 2+ weak signals: 'boring' AND 'waste'; 'bad' AND 'terrible'\n"
                "IMPORTANT: Do NOT only generate keyword-matching functions. Mix strategies.\n"
                "Single common words alone (e.g. 'bad') are OK if combined with metadata constraints."
            ),
            1: (
                "Movie reviews are LONG text. AIM FOR DIVERSE STRATEGIES to maximize coverage.\n"
                "Strategy types (use ALL of these, not just keywords):\n"
                "  1. Multi-word phrases: 'highly recommend', 'must see', 'one of the best'\n"
                "  2. Strong sentiment words: 'masterpiece', 'phenomenal', 'flawless', 'brilliant'\n"
                "  3. Metadata + text combos: e.g., long review (num_words > 200) AND contains 'loved';\n"
                "     many exclamation marks AND contains a positive word\n"
                "  4. Structural patterns: review contains superlative + recommendation;\n"
                "     ALL-CAPS positive words\n"
                "  5. Pattern-based: '(excellent|superb|outstanding) (performance|acting|film)',\n"
                "     'one of the (best|greatest|finest)', 'thoroughly (enjoyed|entertaining)'\n"
                "  6. Rating patterns: '10/10', '10 out of 10', 'five stars', '9/10'\n"
                "  7. Combine 2+ weak signals: 'great' AND 'recommend'; 'love' AND 'amazing'\n"
                "IMPORTANT: Do NOT only generate keyword-matching functions. Mix strategies.\n"
                "Single common words alone (e.g. 'great') are OK if combined with metadata constraints."
            ),
        },
        metadata_fields_description=(
            '  ex["text"]: raw review text (typically 100-1000+ words)\n'
            '  ex["metadata"]["text_normalized"]: normalized text\n'
            '  ex["metadata"]["num_words"], num_chars, num_exclamation, num_question, num_uppercase\n'
            '  ex["metadata"]["has_url"], has_email'
        ),
    ),

    "agnews": TaskConfig(
        task_name="agnews",
        task_description="AG News topic classification (4-class: World, Sports, Business, Sci/Tech).",
        label_names={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        num_classes=4,
        valid_labels={0, 1, 2, 3},
        bool_features=["has_url", "has_email"],
        class_guardrails={
            0: (
                "World news covers international politics, conflicts, diplomacy, and global events.\n"
                "Strategy types (use diverse approaches):\n"
                "  1. Entity-based: country names, regions, international orgs (UN, NATO, EU, African Union)\n"
                "  2. Topic keywords: 'war', 'troops', 'election', 'president', 'minister', 'diplomat',\n"
                "     'hostage', 'rebel', 'peace', 'treaty', 'bomb', 'attack', 'refugee', 'humanitarian'\n"
                "  3. Pattern-based: '(prime minister|president) of', 'talks between .* and',\n"
                "     'UN (Security Council|General Assembly)', 'war in', 'conflict in'\n"
                "  4. Exclusion-aware: many World keywords also appear in Business/Sports.\n"
                "     Combine with exclusion: e.g., contains 'Iraq' but NOT 'stock' or 'game'\n"
                "IMPORTANT: Short text (~40 words). Single distinctive keywords can work well here."
            ),
            1: (
                "Sports news covers games, matches, athletes, teams, scores, and competitions.\n"
                "Strategy types (use diverse approaches):\n"
                "  1. Sport names: 'football', 'soccer', 'baseball', 'basketball', 'tennis', 'golf',\n"
                "     'hockey', 'cricket', 'rugby', 'boxing', 'racing', 'swimming', 'Olympics'\n"
                "  2. Game terms: 'goal', 'score', 'win', 'defeat', 'championship', 'league', 'cup',\n"
                "     'tournament', 'playoff', 'season', 'coach', 'team', 'player', 'match', 'game'\n"
                "  3. Pattern-based: 'beat .* [0-9]+-[0-9]+', 'scored [0-9]+ (goals|points|runs)',\n"
                "     '(won|lost|tied) [0-9]+-[0-9]+', 'vs\\.?\\s', 'World Cup', 'Super Bowl'\n"
                "  4. Entity-based: team names, athlete names, league names (NFL, NBA, MLB, FIFA, ATP)\n"
                "IMPORTANT: Short text (~40 words). Sports vocabulary is very distinctive."
            ),
            2: (
                "Business news covers markets, companies, finance, economy, and corporate events.\n"
                "Strategy types (use diverse approaches):\n"
                "  1. Market terms: 'stock', 'shares', 'market', 'profit', 'revenue', 'earnings',\n"
                "     'billion', 'million', 'investor', 'IPO', 'merger', 'acquisition', 'dividend'\n"
                "  2. Financial entities: 'Wall Street', 'NYSE', 'Nasdaq', 'S&P', 'Dow', 'Fed',\n"
                "     'SEC', 'GDP', 'oil prices', 'interest rate'\n"
                "  3. Pattern-based: '\\$[0-9]', 'percent', 'quarter(ly)?', 'fiscal',\n"
                "     '(CEO|CFO|chairman) of', 'Inc\\.', 'Corp\\.', 'Ltd\\.'\n"
                "  4. Corporate actions: 'filed (for|bankruptcy)', 'announced (layoffs|merger)',\n"
                "     'reported (earnings|profit|loss)', 'agreed to (buy|acquire|merge)'\n"
                "IMPORTANT: Short text (~40 words). Dollar amounts and corporate suffixes are strong signals."
            ),
            3: (
                "Sci/Tech news covers technology, software, hardware, science, research, and internet.\n"
                "Strategy types (use diverse approaches):\n"
                "  1. Tech terms: 'software', 'computer', 'internet', 'website', 'browser', 'server',\n"
                "     'virus', 'hacker', 'download', 'wireless', 'broadband', 'chip', 'processor'\n"
                "  2. Companies/products: 'Microsoft', 'Google', 'Apple', 'Intel', 'Linux', 'Windows',\n"
                "     'Firefox', 'iPod', 'Dell', 'IBM', 'AMD', 'Java', 'Oracle'\n"
                "  3. Science terms: 'research', 'scientists', 'study', 'researchers', 'discovery',\n"
                "     'NASA', 'space', 'Mars', 'genome', 'stem cell', 'species', 'planet'\n"
                "  4. Pattern-based: 'vulnerability in', 'patch for', 'version [0-9]',\n"
                "     'open.source', 'operating system', 'search engine'\n"
                "IMPORTANT: Short text (~40 words). Tech/science vocabulary is highly distinctive.\n"
                "Watch out: some tech company news overlaps with Business (stock prices, earnings).\n"
                "Focus on product/technology content, not financial aspects."
            ),
        },
        metadata_fields_description=(
            '  ex["text"]: news article (title + description)\n'
            '  ex["metadata"]["text_normalized"]: normalized text\n'
            '  ex["metadata"]["num_words"], num_chars, num_exclamation, num_question, num_uppercase\n'
            '  ex["metadata"]["has_url"], has_email'
        ),
    ),

    "chemprot": TaskConfig(
        task_name="chemprot",
        task_description=(
            "ChemProt chemical-protein relation extraction (10-class biomedical).\n"
            "Given a sentence describing a chemical-protein interaction, classify the relation type.\n"
            "Each example contains two highlighted entities (a chemical and a protein) and the sentence.\n"
            "The labeling function should analyze the sentence to determine the relation between the entities."
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
            '     scripts/enrich_fever_metadata.py; use via .get(key, default) so LFs\n'
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

    "vitaminc": TaskConfig(
        task_name="vitaminc",
        task_description=(
            "VitaminC contrastive fact verification (3-class: Supports, Refutes, NotEnoughInfo).\n"
            "Given a claim and evidence from Wikipedia revisions, determine if the evidence supports,\n"
            "refutes, or is not enough to verify the claim.\n"
            "Key challenge: contrastive pairs where small evidence changes flip the label."
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
                "VitaminC has CONTRASTIVE examples — tiny changes flip labels.\n"
                "USE PRE-COMPUTED VERIFICATION FEATURES:\n"
                "  1. High claim_in_evidence (>0.5) + numbers_match=True → values agree\n"
                "  2. High entity_containment (>0.5) + negation_mismatch=False\n"
                "  3. High token_overlap (>0.4) + antonym_present=False\n"
                "  4. Combine: claim_in_evidence > 0.6 and not negation_mismatch and not numbers_mismatch\n"
                "CRITICAL: VitaminC needs PRECISE thresholds. Small numeric differences flip labels.\n"
                "  if meta['claim_in_evidence'] > 0.5 and meta['numbers_match'] and not meta['negation_mismatch']: return 0"
            ),
            1: (
                "Refutes: The evidence CONTRADICTS the claim.\n"
                "VitaminC contrastive: evidence differs by ONE number or word.\n"
                "USE PRE-COMPUTED VERIFICATION FEATURES:\n"
                "  1. numbers_mismatch=True + high entity_containment → same topic, different numbers\n"
                "  2. negation_mismatch=True + high token_overlap → same content, negation flipped\n"
                "  3. antonym_present=True + entity_overlap > 0.3 → same entities, opposite meaning\n"
                "  4. Combine: entity_containment > 0.3 and (numbers_mismatch or negation_mismatch)\n"
                "CRITICAL: Refutation = high topic overlap + specific contradiction signal."
            ),
            2: (
                "NotEnoughInfo: Evidence is INSUFFICIENT to verify or refute.\n"
                "USE PRE-COMPUTED VERIFICATION FEATURES:\n"
                "  1. Low token_overlap (<0.2) → different topics entirely\n"
                "  2. Low entity_containment (<0.2) → claim entities not in evidence\n"
                "  3. num_shared_entities == 0 → no common entities\n"
                "  4. has_numbers_claim=True but has_numbers_evidence=False → missing key data\n"
                "NEI is MINORITY class in VitaminC. Be precise.\n"
                "  if meta['token_overlap'] < 0.15 and meta['entity_containment'] < 0.2: return 2"
            ),
        },
        metadata_fields_description=(
            '  ex["text"]: "Claim: {claim}\\nEvidence: {evidence}" (combined)\n'
            '  ex["metadata"]["claim"]: the claim to be verified\n'
            '  ex["metadata"]["evidence"]: Wikipedia evidence sentence\n'
            '  ex["metadata"]["page"]: Wikipedia page title\n'
            '  ex["metadata"]["revision_type"]: "real" or "synthetic"\n'
            '  ## PRE-COMPUTED VERIFICATION FEATURES (use these!):\n'
            '  ex["metadata"]["token_overlap"]: float, Jaccard of content words (0-1)\n'
            '  ex["metadata"]["claim_in_evidence"]: float, fraction of claim words in evidence (0-1)\n'
            '  ex["metadata"]["evidence_in_claim"]: float, fraction of evidence words in claim (0-1)\n'
            '  ex["metadata"]["entity_overlap"]: float, Jaccard of entity mentions (0-1)\n'
            '  ex["metadata"]["entity_containment"]: float, fraction of claim entities in evidence (0-1)\n'
            '  ex["metadata"]["num_shared_entities"]: int\n'
            '  ex["metadata"]["num_claim_entities"]: int\n'
            '  ex["metadata"]["has_numbers_both"]: bool\n'
            '  ex["metadata"]["numbers_match"]: bool, at least one number matches\n'
            '  ex["metadata"]["numbers_mismatch"]: bool, both have numbers but none match\n'
            '  ex["metadata"]["negation_in_claim"]: bool\n'
            '  ex["metadata"]["negation_in_evidence"]: bool\n'
            '  ex["metadata"]["negation_mismatch"]: bool, one side has negation other does not\n'
            '  ex["metadata"]["antonym_present"]: bool\n'
            '  ex["metadata"]["claim_word_count"]: int\n'
            '  ex["metadata"]["evidence_word_count"]: int\n'
            '  ex["metadata"]["length_ratio"]: float\n'
            '  ex["metadata"]["text_normalized"]: normalized text\n'
            '  ex["metadata"]["has_url"], has_email'
        ),
    ),
    "claude9": TaskConfig(
        task_name="claude9",
        task_description=(
            "Claude9 legal Terms-of-Service clause classification (9-class).\n"
            "Each example is a clause excerpted from a real Terms of Service document.\n"
            "Classify into one of 9 categories: 8 specific clause types or 'Other' (catch-all).\n"
            "Class distribution is highly imbalanced — 'Other' (label 8) dominates ~90% of all splits."
        ),
        label_names={
            0: "Limitation of liability",
            1: "Unilateral termination",
            2: "Unilateral change",
            3: "Content removal",
            4: "Contract by using",
            5: "Choice of law",
            6: "Jurisdiction",
            7: "Arbitration",
            8: "Other",
        },
        num_classes=9,
        valid_labels={0, 1, 2, 3, 4, 5, 6, 7, 8},
        bool_features=[],
        class_guardrails={
            0: ("Limitation-of-liability LFs should look for: 'as is', 'no warranty', "
                "'not liable', 'merchantability', 'fit for purpose', 'indemnify'."),
            1: ("Unilateral-termination LFs should look for: 'terminate at any time', "
                "'sole discretion', 'without notice', 'discontinue', 'suspend'."),
            2: ("Unilateral-change LFs should look for: 'modify these terms', 'amend', "
                "'update' COMBINED with 'at any time' or 'without notice'."),
            3: ("Content-removal LFs should look for: 'remove content', 'take down', "
                "'delete user content', 'screen', 'monitor and remove'."),
            4: ("Contract-by-using LFs should look for: 'by using', 'by accessing', "
                "'you agree', 'constitutes acceptance', 'deemed to have accepted'."),
            5: ("Choice-of-law LFs should look for: 'governed by laws of', "
                "'in accordance with the laws of <state>'."),
            6: ("Jurisdiction LFs should look for: 'exclusive jurisdiction', "
                "'courts of', 'venue', 'submit to the jurisdiction'."),
            7: ("Arbitration LFs should look for: 'arbitration', 'binding arbitration', "
                "'AAA', 'JAMS', 'class action waiver'."),
            8: ("'Other' (label 8) is a catch-all (~90% of examples). Avoid LFs that "
                "default to class 8; prefer ABSTAIN for ambiguous clauses, and reserve "
                "high-precision class 0-7 LFs."),
        },
        metadata_fields_description=(
            '  ex["text"]: legal clause text (lowercased, English)\n'
            '  ex["metadata"]: minimal — Claude9 has only source + weak_labels fields'
        ),
    ),

    "banking77": TaskConfig(
        task_name="banking77",
        task_description=(
            "Banking77 customer-service intent classification (77-class).\n"
            "Each text is a short customer utterance to a banking app/chatbot. Classify the intent.\n"
            "The 77 intents cover: card management (arrival, activation, expiry, blocking, loss),\n"
            "top-ups (by card, bank transfer, cash/cheque; failures, limits, charges, reverts),\n"
            "transfers (timing, fee, decline, failed, not received), refunds/charges/transactions,\n"
            "identity verification, exchange rates, virtual/disposable/spare cards, and account actions.\n"
            "LFs should match distinctive intent keywords/phrases per class. With 77 classes the\n"
            "average prior is ~1.3% so high-precision narrow patterns are required; broad patterns\n"
            "will overlap many classes and hurt precision. Prefer ABSTAIN over guessing."
        ),
        label_names={
            0: "Refund_not_showing_up", 1: "activate_my_card", 2: "age_limit",
            3: "apple_pay_or_google_pay", 4: "atm_support", 5: "automatic_top_up",
            6: "balance_not_updated_after_bank_transfer",
            7: "balance_not_updated_after_cheque_or_cash_deposit",
            8: "beneficiary_not_allowed", 9: "cancel_transfer",
            10: "card_about_to_expire", 11: "card_acceptance", 12: "card_arrival",
            13: "card_delivery_estimate", 14: "card_linking", 15: "card_not_working",
            16: "card_payment_fee_charged", 17: "card_payment_not_recognised",
            18: "card_payment_wrong_exchange_rate", 19: "card_swallowed",
            20: "cash_withdrawal_charge", 21: "cash_withdrawal_not_recognised",
            22: "change_pin", 23: "compromised_card", 24: "contactless_not_working",
            25: "country_support", 26: "declined_card_payment", 27: "declined_cash_withdrawal",
            28: "declined_transfer", 29: "direct_debit_payment_not_recognised",
            30: "disposable_card_limits", 31: "edit_personal_details",
            32: "exchange_charge", 33: "exchange_rate", 34: "exchange_via_app",
            35: "extra_charge_on_statement", 36: "failed_transfer",
            37: "fiat_currency_support", 38: "get_disposable_virtual_card",
            39: "get_physical_card", 40: "getting_spare_card",
            41: "getting_virtual_card", 42: "lost_or_stolen_card",
            43: "lost_or_stolen_phone", 44: "order_physical_card",
            45: "passcode_forgotten", 46: "pending_card_payment",
            47: "pending_cash_withdrawal", 48: "pending_top_up",
            49: "pending_transfer", 50: "pin_blocked", 51: "receiving_money",
            52: "request_refund", 53: "reverted_card_payment?",
            54: "supported_cards_and_currencies", 55: "terminate_account",
            56: "top_up_by_bank_transfer_charge", 57: "top_up_by_card_charge",
            58: "top_up_by_cash_or_cheque", 59: "top_up_failed",
            60: "top_up_limits", 61: "top_up_reverted", 62: "topping_up_by_card",
            63: "transaction_charged_twice", 64: "transfer_fee_charged",
            65: "transfer_into_account", 66: "transfer_not_received_by_recipient",
            67: "transfer_timing", 68: "unable_to_verify_identity",
            69: "verify_my_identity", 70: "verify_source_of_funds",
            71: "verify_top_up", 72: "virtual_card_not_working",
            73: "visa_or_mastercard", 74: "why_verify_identity",
            75: "wrong_amount_of_cash_received",
            76: "wrong_exchange_rate_for_cash_withdrawal",
        },
        num_classes=77,
        valid_labels=set(range(77)),
        bool_features=[],
        # Compact 1-line per-class anchors. With 77 classes per-class guardrails
        # must be short or the Improver prompt blows up the context budget.
        class_guardrails={
            0:  "Refund_not_showing_up: refund missing/late. Keywords: 'refund', 'haven\\'t received', 'still waiting for refund', 'refund not credited'.",
            1:  "activate_my_card: activating a new card. Keywords: 'activate', 'activation', 'how do I activate', 'enable my card'.",
            2:  "age_limit: minimum age to open account. Keywords: 'age limit', 'how old', 'minimum age', 'underage', 'too young'.",
            3:  "apple_pay_or_google_pay: mobile-wallet support. Keywords: 'Apple Pay', 'Google Pay', 'wallet', 'add to phone', 'NFC payment'.",
            4:  "atm_support: ATM-network compatibility. Keywords: 'ATM', 'cash machine', 'which ATMs', 'ATM network', 'use abroad ATM'.",
            5:  "automatic_top_up: auto-top-up setup. Keywords: 'automatic top up', 'auto top-up', 'recurring top up', 'top up automatically'.",
            6:  "balance_not_updated_after_bank_transfer: deposit by transfer missing. Keywords: 'balance not updated', 'transferred but balance', 'bank transfer not showing'.",
            7:  "balance_not_updated_after_cheque_or_cash_deposit: deposit by cheque/cash missing. Keywords: 'deposited cheque', 'cash deposit', 'balance not updated', 'cheque not showing'.",
            8:  "beneficiary_not_allowed: recipient blocked. Keywords: 'beneficiary not allowed', 'cannot add recipient', 'beneficiary rejected'.",
            9:  "cancel_transfer: cancel a pending/sent transfer. Keywords: 'cancel transfer', 'stop transfer', 'reverse transfer', 'cancel payment'.",
            10: "card_about_to_expire: expiry imminent. Keywords: 'card expires', 'card expiring', 'about to expire', 'new card before expiry'.",
            11: "card_acceptance: where the card is accepted. Keywords: 'where can I use', 'is my card accepted', 'merchants accept'.",
            12: "card_arrival: when ordered card arrives. Keywords: 'when will my card arrive', 'still waiting for card', 'card not received yet'.",
            13: "card_delivery_estimate: shipping ETA. Keywords: 'delivery estimate', 'how long delivery', 'shipping time', 'when posted'.",
            14: "card_linking: link to existing account. Keywords: 'link card', 'connect card', 'add card to account'.",
            15: "card_not_working: payment fails at point of sale. Keywords: 'card not working', 'card declined', 'card stopped working', 'card error'.",
            16: "card_payment_fee_charged: fee on payment. Keywords: 'fee on payment', 'charged a fee for payment', 'card payment fee'.",
            17: "card_payment_not_recognised: unknown card charge. Keywords: 'don\\'t recognise', 'didn\\'t make this payment', 'unknown charge'.",
            18: "card_payment_wrong_exchange_rate: bad FX on card payment. Keywords: 'wrong exchange rate', 'incorrect exchange rate', 'rate not matching'.",
            19: "card_swallowed: ATM ate the card. Keywords: 'ATM swallowed', 'card stuck in ATM', 'machine took my card'.",
            20: "cash_withdrawal_charge: ATM withdrawal fee. Keywords: 'cash withdrawal fee', 'charge for withdrawing', 'ATM fee'.",
            21: "cash_withdrawal_not_recognised: unknown ATM withdrawal. Keywords: 'don\\'t recognise withdrawal', 'didn\\'t withdraw', 'unknown ATM transaction'.",
            22: "change_pin: change the PIN. Keywords: 'change PIN', 'reset PIN', 'new PIN', 'update PIN'.",
            23: "compromised_card: fraud suspected. Keywords: 'compromised', 'stolen details', 'fraud', 'hacked card', 'someone has my card'.",
            24: "contactless_not_working: tap-to-pay fails. Keywords: 'contactless not working', 'tap doesn\\'t work', 'NFC not working'.",
            25: "country_support: country availability. Keywords: 'can I use in', 'supported in country', 'available in <country>'.",
            26: "declined_card_payment: card payment declined. Keywords: 'payment declined', 'transaction declined', 'declined at checkout'.",
            27: "declined_cash_withdrawal: ATM withdrawal declined. Keywords: 'cash withdrawal declined', 'ATM declined', 'couldn\\'t withdraw'.",
            28: "declined_transfer: bank transfer declined. Keywords: 'transfer declined', 'transfer rejected', 'wouldn\\'t go through'.",
            29: "direct_debit_payment_not_recognised: unknown direct debit. Keywords: 'direct debit', 'didn\\'t set up', 'unknown DD', 'unrecognised direct debit'.",
            30: "disposable_card_limits: limits on virtual disposable cards. Keywords: 'disposable card limit', 'virtual card max', 'limit on disposable'.",
            31: "edit_personal_details: update name/address/etc. Keywords: 'change address', 'update name', 'edit personal details', 'change phone number'.",
            32: "exchange_charge: fee for FX. Keywords: 'exchange charge', 'fee for exchange', 'currency conversion fee'.",
            33: "exchange_rate: which FX rate. Keywords: 'exchange rate', 'what rate', 'conversion rate', 'today\\'s rate'.",
            34: "exchange_via_app: exchange currency in app. Keywords: 'exchange in app', 'convert currency', 'swap currency in app'.",
            35: "extra_charge_on_statement: unexpected line item. Keywords: 'extra charge', 'why was I charged', 'unknown fee on statement'.",
            36: "failed_transfer: transfer failed. Keywords: 'transfer failed', 'payment failed', 'didn\\'t go through'.",
            37: "fiat_currency_support: which currencies. Keywords: 'supported currencies', 'do you support', 'currencies available'.",
            38: "get_disposable_virtual_card: how to get disposable virtual card. Keywords: 'disposable virtual card', 'one-time card', 'temporary virtual'.",
            39: "get_physical_card: order physical card. Keywords: 'physical card', 'get a real card', 'plastic card'.",
            40: "getting_spare_card: order spare card. Keywords: 'spare card', 'additional card', 'second card'.",
            41: "getting_virtual_card: order virtual card. Keywords: 'virtual card', 'digital card', 'card in app'.",
            42: "lost_or_stolen_card: card lost or stolen. Keywords: 'lost my card', 'stolen card', 'card was stolen', 'misplaced card'.",
            43: "lost_or_stolen_phone: phone with banking app lost/stolen. Keywords: 'lost my phone', 'phone stolen', 'lost device'.",
            44: "order_physical_card: order new physical card. Keywords: 'order a card', 'request a physical card', 'send me a card'.",
            45: "passcode_forgotten: forgot login passcode. Keywords: 'forgot passcode', 'forgot password', 'can\\'t log in', 'reset passcode'.",
            46: "pending_card_payment: card payment pending. Keywords: 'pending card payment', 'still pending', 'card payment not cleared'.",
            47: "pending_cash_withdrawal: ATM withdrawal pending. Keywords: 'pending withdrawal', 'cash withdrawal pending'.",
            48: "pending_top_up: top-up pending. Keywords: 'pending top up', 'top up not gone through yet', 'still pending'.",
            49: "pending_transfer: transfer pending. Keywords: 'pending transfer', 'transfer still pending', 'not cleared'.",
            50: "pin_blocked: PIN locked after wrong tries. Keywords: 'PIN blocked', 'PIN locked', 'too many attempts', 'wrong PIN locked'.",
            51: "receiving_money: how to receive money. Keywords: 'receive money', 'get money sent to me', 'incoming transfer'.",
            52: "request_refund: want a refund. Keywords: 'I want a refund', 'request refund', 'refund please', 'how to get refund'.",
            53: "reverted_card_payment?: payment got reverted. Keywords: 'payment reverted', 'reversed payment', 'money came back'.",
            54: "supported_cards_and_currencies: which cards/currencies. Keywords: 'cards supported', 'currencies supported', 'compatible cards'.",
            55: "terminate_account: close account. Keywords: 'close account', 'terminate account', 'delete my account', 'cancel account'.",
            56: "top_up_by_bank_transfer_charge: fee for top-up via transfer. Keywords: 'top up by bank transfer fee', 'charge for top up by transfer'.",
            57: "top_up_by_card_charge: fee for top-up via card. Keywords: 'top up by card fee', 'fee top up card'.",
            58: "top_up_by_cash_or_cheque: top-up via cash/cheque. Keywords: 'top up by cash', 'top up by cheque', 'deposit cash to top up'.",
            59: "top_up_failed: top-up failed. Keywords: 'top up failed', 'top up didn\\'t work', 'top up declined'.",
            60: "top_up_limits: max top-up amount. Keywords: 'top up limit', 'maximum top up', 'how much can I top up'.",
            61: "top_up_reverted: top-up was reverted. Keywords: 'top up reverted', 'top up returned', 'money came back'.",
            62: "topping_up_by_card: how to top up via card. Keywords: 'top up by card', 'how to top up using card', 'top up with debit card'.",
            63: "transaction_charged_twice: duplicate charge. Keywords: 'charged twice', 'duplicate charge', 'paid twice'.",
            64: "transfer_fee_charged: fee on a transfer. Keywords: 'transfer fee', 'charged for transfer', 'fee on transfer'.",
            65: "transfer_into_account: how to transfer into account. Keywords: 'transfer to my account', 'send money to account', 'add money via transfer'.",
            66: "transfer_not_received_by_recipient: recipient didn\\'t get it. Keywords: 'recipient didn\\'t receive', 'they didn\\'t get the money', 'transfer not arrived'.",
            67: "transfer_timing: how long transfer takes. Keywords: 'how long transfer', 'when will it arrive', 'transfer time'.",
            68: "unable_to_verify_identity: verification failed. Keywords: 'unable to verify', 'verification failed', 'can\\'t verify my identity'.",
            69: "verify_my_identity: how to verify. Keywords: 'verify my identity', 'identity verification', 'KYC', 'how do I verify'.",
            70: "verify_source_of_funds: source-of-funds check. Keywords: 'source of funds', 'verify source', 'where the money came from'.",
            71: "verify_top_up: verify a top-up source. Keywords: 'verify top up', 'top up verification'.",
            72: "virtual_card_not_working: virtual card fails. Keywords: 'virtual card not working', 'virtual card declined', 'digital card error'.",
            73: "visa_or_mastercard: which network. Keywords: 'visa or mastercard', 'which network', 'is it visa', 'is it mastercard'.",
            74: "why_verify_identity: reason for verification. Keywords: 'why do I need to verify', 'why verify identity', 'why KYC'.",
            75: "wrong_amount_of_cash_received: ATM gave wrong cash. Keywords: 'wrong amount of cash', 'ATM gave less', 'incorrect cash received'.",
            76: "wrong_exchange_rate_for_cash_withdrawal: FX wrong at ATM. Keywords: 'wrong exchange rate at ATM', 'incorrect rate cash withdrawal'.",
        },
        metadata_fields_description=(
            '  ex["text"]: short customer utterance (10-30 words typical)\n'
            '  ex["metadata"]: only "source" — no pre-computed features for Banking77'
        ),
    ),

    "ddi": TaskConfig(
        task_name="ddi",
        task_description=(
            "DDI drug-drug interaction relation extraction (5-class biomedical).\n"
            "Given a biomedical sentence with two highlighted drug entities, classify the interaction type.\n"
            "Each example contains two drugs (entity1, entity2) and the sentence between/around them.\n"
            "The class distribution is heavily skewed toward 'no_interaction' (~78%); precision-floor LFs\n"
            "should focus on the 4 minority classes (mechanism / effect / advise / int)."
        ),
        label_names={
            0: "no_interaction",
            1: "mechanism",
            2: "effect",
            3: "advise",
            4: "int",
        },
        num_classes=5,
        valid_labels={0, 1, 2, 3, 4},
        bool_features=[],
        class_guardrails={
            0: (
                "no_interaction: The sentence does NOT describe a clinically relevant interaction.\n"
                "Most train examples (~78%); do NOT default to class 0 unless explicit no-effect language.\n"
                "Keywords: 'no interaction', 'no effect', 'did not affect', 'no significant',\n"
                "  'no clinically relevant', 'no change', 'no statistically significant'\n"
                "NOTE: Prefer ABSTAIN over guessing class 0; the precision floor will reject defaults."
            ),
            1: (
                "mechanism: Pharmacokinetic interaction at the mechanism level (metabolism / induction / inhibition / absorption).\n"
                "Keywords: 'metabolism', 'metabolized by', 'CYP3A4', 'CYP', 'inducer', 'inducing',\n"
                "  'inhibitor of', 'enzyme', 'P-glycoprotein', 'absorption', 'clearance',\n"
                "  'plasma concentration', 'AUC', 'Cmax', 'half-life', 'pharmacokinetic'\n"
                "Pattern: 'A increased AUC of B', 'A is metabolized by CYP3A4', 'A induces CYP'\n"
                "NOTE: Look for enzyme names + pharmacokinetic vocabulary."
            ),
            2: (
                "effect: Pharmacodynamic interaction at the effect level (the observed clinical consequence).\n"
                "Keywords: 'increased risk', 'enhanced effect', 'potentiates', 'additive', 'synergistic',\n"
                "  'antagonize', 'reduces effect', 'attenuates', 'augments', 'bleeding', 'toxicity',\n"
                "  'hypotension', 'sedation', 'CNS depression'\n"
                "Pattern: 'A potentiates the effect of B', 'increased risk of X when A + B'\n"
                "NOTE: Effect-level = clinical outcomes; distinguish from mechanism (PK-level)."
            ),
            3: (
                "advise: Clinical advice / recommendation about co-administration.\n"
                "Keywords: 'should be avoided', 'should not be used', 'caution', 'concomitant use',\n"
                "  'use with caution', 'not recommended', 'monitor closely', 'dose adjustment',\n"
                "  'contraindicated', 'avoid concomitant', 'careful monitoring'\n"
                "Pattern: 'A and B should not be administered concomitantly', 'caution when used with'\n"
                "NOTE: Advise = recommendation language, often imperative ('should', 'must', 'avoid')."
            ),
            4: (
                "int: General interaction stated but not specifying mechanism / effect / advise.\n"
                "Keywords: 'interaction', 'interact with', 'interacts', 'drug interaction',\n"
                "  'potential interaction'\n"
                "Pattern: 'A interacts with B' (with no specific mechanism / effect / advice nearby)\n"
                "NOTE: Rare class (~1% of train). Only triggers when the sentence states interaction\n"
                "  exists without elaborating mechanism / effect / advice. High precision required."
            ),
        },
        metadata_fields_description=(
            '  ex["text"]: biomedical sentence describing drug-drug interaction\n'
            '  ex["metadata"]["text_normalized"]: lowercased normalized text\n'
            '  ex["metadata"]["entity1"]: first drug name\n'
            '  ex["metadata"]["entity2"]: second drug name\n'
            '  ex["metadata"]["doc_id"]: source DrugBank / MedLine document id'
        ),
    ),

    "scifact": TaskConfig(
        task_name="scifact",
        task_description=(
            "SciFact scientific claim verification (3-class: Supports, Contradicts, NotEnoughInfo).\n"
            "Given a scientific claim and evidence from a biomedical / scientific abstract, decide if\n"
            "the evidence supports, contradicts, or is insufficient to verify the claim.\n"
            "Scientific writing makes the task harder than FEVER (more technical vocabulary, numeric\n"
            "evidence, hedged language). LFs should match scientific-paper cues."
        ),
        label_names={0: "Supports", 1: "Contradicts", 2: "NotEnoughInfo"},
        num_classes=3,
        valid_labels={0, 1, 2},
        bool_features=[
            "has_numbers_claim", "has_numbers_evidence", "has_numbers_both",
            "numbers_match", "numbers_mismatch",
            "negation_in_claim", "negation_in_evidence", "negation_mismatch",
            "antonym_present",
        ],
        class_guardrails={
            0: (
                "Supports: The scientific evidence CONFIRMS the claim.\n"
                "Strong cues:\n"
                "  1. Evidence's findings agree with the claim's polarity (no negation_mismatch)\n"
                "  2. Numeric statements match (numbers_match=True, numbers_mismatch=False)\n"
                "  3. High lexical overlap of claim entities/terms in evidence\n"
                "  4. Evidence phrases like 'we show', 'we demonstrate', 'our results confirm',\n"
                "     'consistent with', 'we found that', 'significantly increased'\n"
                "  5. No hedging language (avoid 'may', 'might', 'could' as supports cues)\n"
                "Combine: if claim_in_evidence > 0.5 and not negation_mismatch and (numbers_match or no numbers): return 0"
            ),
            1: (
                "Contradicts: The evidence DISAGREES with the claim.\n"
                "Strong cues:\n"
                "  1. negation_mismatch=True (one side negates, other does not)\n"
                "  2. numbers_mismatch=True (both have numbers and they disagree)\n"
                "  3. antonym_present=True (e.g., claim 'increases' vs evidence 'decreases')\n"
                "  4. Evidence phrases like 'we did not find', 'no significant', 'failed to show',\n"
                "     'contrary to', 'inconsistent with', 'not associated', 'no evidence for'\n"
                "  5. Direction-of-effect opposite: claim says 'increase' but evidence says 'decrease'\n"
                "Combine: if entity_containment > 0.3 and (negation_mismatch or numbers_mismatch or antonym_present): return 1"
            ),
            2: (
                "NotEnoughInfo: The evidence is INSUFFICIENT to verify or refute (off-topic or hedged).\n"
                "Strong cues:\n"
                "  1. Low entity overlap (claim's entities not in evidence)\n"
                "  2. Evidence discusses a related but different question\n"
                "  3. Hedging language: 'may', 'might', 'possibly', 'further research needed',\n"
                "     'remains unclear', 'limited evidence', 'we suggest'\n"
                "  4. Evidence is from an unrelated subfield (different organism, different mechanism)\n"
                "Combine: if entity_overlap < 0.2 and not negation_mismatch: return 2 (likely NEI)"
            ),
        },
        metadata_fields_description=(
            '  ex["text"]: "Claim: {claim}\\nEvidence: {evidence}" (combined)\n'
            '  ex["metadata"]["claim"]: the scientific claim to be verified\n'
            '  ex["metadata"]["cited_doc_ids"]: list of source document ids\n'
            '  ## PRE-COMPUTED FEATURES (present after enrichment; use .get(key, default)):\n'
            '  ex["metadata"].get("token_overlap", 0.0): float, Jaccard of content words\n'
            '  ex["metadata"].get("claim_in_evidence", 0.0): float, fraction of claim words in evidence\n'
            '  ex["metadata"].get("entity_overlap", 0.0): float, Jaccard of entity mentions\n'
            '  ex["metadata"].get("entity_containment", 0.0): float, fraction of claim entities in evidence\n'
            '  ex["metadata"].get("numbers_match", False): bool\n'
            '  ex["metadata"].get("numbers_mismatch", False): bool\n'
            '  ex["metadata"].get("negation_mismatch", False): bool\n'
            '  ex["metadata"].get("antonym_present", False): bool\n'
            '  ## OPTIONAL after running scripts/enrich_fever_metadata.py with --input_dir data/processed/scifact:\n'
            '  ex["metadata"].get("nli_contradict", 0.0): float (0-1)\n'
            '    -> >0.5 => strong Contradicts signal\n'
            '  ex["metadata"].get("nli_entail", 0.0): float (0-1)\n'
            '    -> >0.6 => Supports\n'
            '  ex["metadata"].get("refutes_phrase_count", 0): int, hits on refutation phrase bank\n'
            '  ex["metadata"].get("supports_phrase_count", 0): int, hits on supports phrase bank'
        ),
    ),

    # ─── Ohsumed-ML (23-class multi-label biomedical literature) ─────────
    # MULTI-LABEL: each doc has true_labels: List[int] (avg 1.66 labels/doc).
    # Generator authors per-class LFs that return one class; multi-label
    # aggregation (ml_run_annotators) combines votes from multiple LFs.
    "ohsumed_ml": TaskConfig(
        task_name="ohsumed_ml",
        task_description=(
            "Ohsumed multi-label biomedical literature classification (23 MeSH C-categories).\n"
            "Given a PubMed/MEDLINE abstract, identify ALL relevant disease categories.\n"
            "Each doc may carry MULTIPLE labels (avg 1.66 per doc, max 8). Write LFs\n"
            "that fire on patterns SPECIFIC to one C-category — multiple LFs firing\n"
            "different categories on the same abstract is expected and wanted."
        ),
        label_names={
            0:  "C01_Bacterial_Infections_and_Mycoses",
            1:  "C02_Virus_Diseases",
            2:  "C03_Parasitic_Diseases",
            3:  "C04_Neoplasms",
            4:  "C05_Musculoskeletal_Diseases",
            5:  "C06_Digestive_System_Diseases",
            6:  "C07_Stomatognathic_Diseases",
            7:  "C08_Respiratory_Tract_Diseases",
            8:  "C09_Otorhinolaryngologic_Diseases",
            9:  "C10_Nervous_System_Diseases",
            10: "C11_Eye_Diseases",
            11: "C12_Urologic_and_Male_Genital_Diseases",
            12: "C13_Female_Genital_Diseases_and_Pregnancy",
            13: "C14_Cardiovascular_Diseases",
            14: "C15_Hemic_and_Lymphatic_Diseases",
            15: "C16_Neonatal_Diseases_and_Abnormalities",
            16: "C17_Skin_and_Connective_Tissue_Diseases",
            17: "C18_Nutritional_and_Metabolic_Diseases",
            18: "C19_Endocrine_Diseases",
            19: "C20_Immunologic_Diseases",
            20: "C21_Disorders_of_Environmental_Origin",
            21: "C22_Animal_Diseases",
            22: "C23_Pathological_Conditions_Signs_and_Symptoms",
        },
        num_classes=23,
        valid_labels=set(range(23)),
        bool_features=[],
        class_guardrails={
            0: "C01 Bacterial Infections/Mycoses: bacterial pathogens/mycoses. Keywords: 'staphylococcus','streptococcus','tuberculosis','sepsis','bacterial','fungal infection','candidiasis'.",
            1: "C02 Virus Diseases. Keywords: 'HIV','hepatitis','influenza','herpes','coronavirus','HPV','viral','vaccine'.",
            2: "C03 Parasitic Diseases. Keywords: 'malaria','plasmodium','schistosoma','helminth','parasite','trypanosoma'.",
            3: "C04 Neoplasms (cancer). Keywords: 'cancer','tumor','carcinoma','malignant','metastasis','chemotherapy','oncology','lymphoma'.",
            4: "C05 Musculoskeletal: bone/joint/muscle. Keywords: 'arthritis','osteoporosis','fracture','rheumatoid','bone','muscle','tendon'.",
            5: "C06 Digestive/hepatic. Keywords: 'colitis','Crohn','gastric','liver','hepatic','intestinal','colorectal','cirrhosis','peptic ulcer'.",
            6: "C07 Stomatognathic (oral/dental). Keywords: 'dental','periodontal','tooth','oral','salivary','gingival','TMJ'.",
            7: "C08 Respiratory: lung/airway. Keywords: 'asthma','COPD','pneumonia','tuberculosis','pulmonary','respiratory','bronchitis','dyspnea'.",
            8: "C09 Otorhinolaryngologic (ENT). Keywords: 'otitis','sinusitis','tinnitus','hearing loss','laryngeal','pharyngeal','vestibular'.",
            9: "C10 Nervous: brain/nerve. Keywords: 'Alzheimer','Parkinson','stroke','epilepsy','neurological','multiple sclerosis','seizure','dementia'.",
            10: "C11 Eye (ocular). Keywords: 'retinal','glaucoma','cataract','ocular','visual','myopia','macular','retinopathy'.",
            11: "C12 Urologic/male genital. Keywords: 'renal','kidney','prostate','urinary','dialysis','glomerular','nephritis','hematuria'.",
            12: "C13 Female genital/pregnancy. Keywords: 'pregnancy','obstetric','gynecologic','ovarian','uterine','menstrual','preeclampsia','cesarean'.",
            13: "C14 Cardiovascular. Keywords: 'cardiovascular','coronary','hypertension','myocardial','atherosclerosis','cardiac','arrhythmia','aortic'.",
            14: "C15 Hematologic. Keywords: 'leukemia','lymphoma','anemia','thrombocytopenia','hematologic','platelet','coagulation','hemophilia'.",
            15: "C16 Neonatal/congenital. Keywords: 'neonatal','congenital','infant','prematurity','birth defect','gestational','fetal'.",
            16: "C17 Skin/connective tissue. Keywords: 'dermatologic','psoriasis','eczema','melanoma','cutaneous','collagen','lupus','scleroderma'.",
            17: "C18 Nutritional/metabolic (not endocrine). Keywords: 'obesity','malnutrition','metabolic syndrome','vitamin deficiency','nutritional','BMI'.",
            18: "C19 Endocrine. Keywords: 'diabetes','thyroid','hypothyroidism','pituitary','adrenal','hormone','insulin','endocrine'.",
            19: "C20 Immunologic. Keywords: 'autoimmune','lupus','immunodeficiency','allergy','immune','IgG','cytokine','HLA'.",
            20: "C21 Environmental disorders. Keywords: 'poisoning','toxicity','radiation exposure','occupational','environmental','heavy metal'.",
            21: "C22 Animal Diseases (veterinary). Keywords: 'veterinary','canine','feline','bovine','swine','equine','animal model','murine'.",
            22: "C23 Generic Path Conditions/Signs/Symptoms. Keywords: 'pain','inflammation','fever','edema','fatigue','syndrome','hemorrhage','necrosis'.",
        },
        metadata_fields_description=(
            '  ex["text"]: raw biomedical abstract\n'
            '  ex["true_labels"]: List[int] subset of 23 C-categories (multi-label)\n'
            '  ex["true_label"]: int — first label of true_labels (single-label projection for sampling)\n'
        ),
    ),

    # ─── Kaggle PubMed-ML (14-class multi-label MeSH top-level) ──────────
    "kaggle_pubmed": TaskConfig(
        task_name="kaggle_pubmed",
        task_description=(
            "PubMed multi-label MeSH top-level classification (14 categories: A-N, no K/V).\n"
            "Given a PubMed abstract, identify ALL relevant MeSH top categories.\n"
            "Each doc covers MULTIPLE categories (avg 5.72/14 = dense multi-label).\n"
            "Distribution dominated by B (Organisms, 93%), E (Techniques, 79%),\n"
            "D (Chemicals, 62%); smaller categories (L, Z) sparse. Write LFs specific\n"
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
