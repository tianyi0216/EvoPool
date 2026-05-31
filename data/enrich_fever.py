#!/usr/bin/env python3
"""EvoPool: optional 5-feature enrichment for FEVER metadata.

Augments the per-row metadata dict produced by ``prepare_fever.py`` with extra
fields that annotators can reference (via ``.get(key, default)``). The enriched
JSONL is a STRICT SUPERSET of the base JSONL -- existing keys are never modified.

Usage:
    python -m data.enrich_fever \\
        --input_dir  data/processed/fever \\
        --output_dir data/processed/fever_enriched \\
        --features   A,B,C,D,E

Features (any combination; commas):
    A  semantic : sentence-transformers cosine similarity (claim vs evidence).
                  Adds meta['sim_semantic'] (float) + meta['sim_semantic_bin'] (str).
                  Falls back to TF-IDF cosine if sentence-transformers/torch missing.
    B  nli      : pretrained NLI (DeBERTa-v3-base-mnli-fever-anli) entail/neutral/contradict.
                  Adds meta['nli_entail'], meta['nli_neutral'], meta['nli_contradict'].
    C  deppar   : spaCy dependency parse -> claim SVO + SVO-in-evidence booleans.
                  Adds meta['claim_subj/verb/obj'] + meta['{subj,verb,obj}_in_evidence'].
    D  antonym  : WordNet-based claim<->evidence antonym pair count.
                  Adds meta['antonym_score_ext'] (int).
    E  phrases  : extended refutes/supports/NEI phrase-bank counts.
                  Adds meta['refutes_phrase_count'], meta['supports_phrase_count'],
                  meta['nei_phrase_count'].

Dependencies are loaded lazily per feature so a subset can be enabled without
pulling in the full ML stack.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


FEATURE_NAMES = {
    "A": "semantic",
    "B": "nli",
    "C": "deppar",
    "D": "antonym",
    "E": "phrases",
}


def load_split(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def save_split(path: Path, examples: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


# ---------------------------------------------------------------------------
# Feature A: semantic similarity (sentence-transformers; TF-IDF fallback)
# ---------------------------------------------------------------------------
def _bin_from_quartiles(vals, sims):
    import numpy as np
    q = np.percentile(vals, [25, 50, 75])
    bins = []
    for s in sims:
        if s < q[0]:
            bins.append("very_low")
        elif s < q[1]:
            bins.append("low")
        elif s < q[2]:
            bins.append("med")
        else:
            bins.append("high")
    return bins, q


def add_semantic(examples, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    import numpy as np
    claims = [ex["metadata"]["claim"] for ex in examples]
    evidences = [ex["metadata"]["evidence"] for ex in examples]

    try:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [semantic] backend=sentence-transformers ({model_name}) on {device}", flush=True)
        model = SentenceTransformer(model_name, device=device)
        t0 = time.time()
        c_emb = model.encode(claims, batch_size=64, show_progress_bar=True,
                             convert_to_numpy=True, normalize_embeddings=True)
        e_emb = model.encode(evidences, batch_size=64, show_progress_bar=True,
                             convert_to_numpy=True, normalize_embeddings=True)
        sims = (c_emb * e_emb).sum(axis=1)
        backend = "st_minilm"
        print(f"  [semantic] encoded in {time.time()-t0:.1f}s (sentence-transformers)", flush=True)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  [semantic] sentence-transformers unavailable ({e}); using TF-IDF cosine", flush=True)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        t0 = time.time()
        vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.95,
            sublinear_tf=True, analyzer="word", strip_accents="unicode",
        )
        vec.fit(claims + evidences)
        c_mat = vec.transform(claims)
        e_mat = vec.transform(evidences)
        c_norm = normalize(c_mat, axis=1)
        e_norm = normalize(e_mat, axis=1)
        sims = np.asarray(c_norm.multiply(e_norm).sum(axis=1)).ravel()
        backend = "tfidf_cosine"
        print(f"  [semantic] encoded in {time.time()-t0:.1f}s (tfidf fallback) "
              f"mean={sims.mean():.3f} std={sims.std():.3f}", flush=True)

    bins, q = _bin_from_quartiles(sims, sims)
    print(f"  [semantic] quartile bins ({backend}): "
          f"<{q[0]:.3f}=very_low | <{q[1]:.3f}=low | <{q[2]:.3f}=med | >={q[2]:.3f}=high",
          flush=True)
    for ex, s, b in zip(examples, sims, bins):
        ex["metadata"]["sim_semantic"] = round(float(s), 4)
        ex["metadata"]["sim_semantic_bin"] = b
        ex["metadata"]["sim_semantic_backend"] = backend
    return examples


# ---------------------------------------------------------------------------
# Feature B: NLI entail / neutral / contradict
# ---------------------------------------------------------------------------
def add_nli(examples, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [nli] loading {model_name} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    print(f"  [nli] id2label = {id2label}", flush=True)

    batch_size = 16 if device == "cuda" else 8
    n = len(examples)
    entail_scores = [0.0] * n
    neutral_scores = [0.0] * n
    contra_scores = [0.0] * n
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = examples[i:i + batch_size]
            premises = [ex["metadata"]["evidence"] for ex in batch]
            hypotheses = [ex["metadata"]["claim"] for ex in batch]
            enc = tokenizer(premises, hypotheses, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for j, p in enumerate(probs):
                idx = i + j
                for k, v in id2label.items():
                    if "entail" in v:
                        entail_scores[idx] = float(p[k])
                    elif "neutral" in v:
                        neutral_scores[idx] = float(p[k])
                    elif "contra" in v:
                        contra_scores[idx] = float(p[k])
            if (i // batch_size) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / max(1, i + batch_size) * (n - i - batch_size)
                print(f"    {i+len(batch)}/{n}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    for ex, e, n_, c in zip(examples, entail_scores, neutral_scores, contra_scores):
        ex["metadata"]["nli_entail"] = round(e, 4)
        ex["metadata"]["nli_neutral"] = round(n_, 4)
        ex["metadata"]["nli_contradict"] = round(c, 4)
    print(f"  [nli] done in {time.time()-t0:.1f}s", flush=True)
    return examples


# ---------------------------------------------------------------------------
# Feature C: dependency-parse SVO + overlap
# ---------------------------------------------------------------------------
def add_deppar(examples):
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError:
        print("  [deppar] downloading spaCy en_core_web_sm...", flush=True)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    t0 = time.time()
    claims = [ex["metadata"]["claim"] for ex in examples]
    evidences = [ex["metadata"]["evidence"] for ex in examples]
    for i, (doc_c, ex, evidence) in enumerate(zip(nlp.pipe(claims, batch_size=64),
                                                  examples, evidences)):
        subj = verb = obj = None
        for tok in doc_c:
            if subj is None and tok.dep_ in ("nsubj", "nsubjpass"):
                subj = tok.text.lower()
            if verb is None and tok.pos_ == "VERB":
                verb = tok.lemma_.lower()
            if obj is None and tok.dep_ in ("dobj", "pobj", "attr"):
                obj = tok.text.lower()
        ev_lower = evidence.lower()
        ex["metadata"]["claim_subj"] = subj or ""
        ex["metadata"]["claim_verb"] = verb or ""
        ex["metadata"]["claim_obj"] = obj or ""
        ex["metadata"]["subj_in_evidence"] = bool(subj and subj in ev_lower)
        ex["metadata"]["verb_in_evidence"] = bool(verb and verb in ev_lower)
        ex["metadata"]["obj_in_evidence"] = bool(obj and obj in ev_lower)
        if i and i % 5000 == 0:
            print(f"    parsed {i}/{len(examples)}", flush=True)
    print(f"  [deppar] done in {time.time()-t0:.1f}s", flush=True)
    return examples


# ---------------------------------------------------------------------------
# Feature D: WordNet-based antonym count
# ---------------------------------------------------------------------------
def add_antonym(examples):
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("good")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn

    ant_cache: Dict[str, set] = {}

    def antonyms_of(word: str) -> set:
        w = word.lower()
        if w in ant_cache:
            return ant_cache[w]
        out = set()
        for syn in wn.synsets(w):
            for lemma in syn.lemmas():
                for a in lemma.antonyms():
                    out.add(a.name().lower().replace("_", " "))
        ant_cache[w] = out
        return out

    t0 = time.time()
    for i, ex in enumerate(examples):
        cw = [w for w in ex["metadata"]["claim"].lower().split() if len(w) > 2]
        ev_set = set(w for w in ex["metadata"]["evidence"].lower().split() if len(w) > 2)
        cnt = 0
        for w in cw:
            if antonyms_of(w) & ev_set:
                cnt += 1
        ex["metadata"]["antonym_score_ext"] = cnt
        if i and i % 5000 == 0:
            print(f"    antonym-scored {i}/{len(examples)}", flush=True)
    print(f"  [antonym] done in {time.time()-t0:.1f}s  cache_size={len(ant_cache)}", flush=True)
    return examples


# ---------------------------------------------------------------------------
# Feature E: extended phrase-bank counts
# ---------------------------------------------------------------------------
REFUTES_PHRASES = [
    "not ", "no ", "never", "none", "cannot", "wasn't", "isn't", "weren't",
    "don't", "doesn't", "didn't", "was not", "is not", "were not", "does not",
    "did not", "do not", "none of", "neither ", " nor ", "without", "unable to",
    "incapable", "excluded", "denied", "rejected", "refused", "opposes",
    "contradicts", "disputes", "falsely", "untrue", " false ", "incorrect",
    "inaccurate", "only", "merely", "solely", "exclusively", "rather than",
    "instead of", "lacks", "lacking",
]
SUPPORTS_PHRASES = [
    "confirms", "verifies", "validates", "supports", "affirms",
    "is consistent with", "matches", "agrees with", "in fact", "indeed ",
    "is indeed", "according to", "documents", "proves", "demonstrates",
    "shows that", "established that",
]
NEI_PHRASES = [
    "may be", "might be", "could be", "possibly", "perhaps", "suggests",
    "implies", "believed to", "speculated", "rumored", "allegedly",
    "reportedly", "potentially", "it is unclear", "unclear whether",
    "remains unknown", "cannot be determined",
]


def add_phrases(examples):
    for ex in examples:
        text = ((ex.get("metadata") or {}).get("text_normalized")
                or ex.get("text") or "").lower()
        ex["metadata"]["refutes_phrase_count"] = sum(1 for p in REFUTES_PHRASES if p in text)
        ex["metadata"]["supports_phrase_count"] = sum(1 for p in SUPPORTS_PHRASES if p in text)
        ex["metadata"]["nei_phrase_count"] = sum(1 for p in NEI_PHRASES if p in text)
    print("  [phrases] done", flush=True)
    return examples


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
FEATURE_FNS = {
    "A": add_semantic,
    "B": add_nli,
    "C": add_deppar,
    "D": add_antonym,
    "E": add_phrases,
}


def enrich_fever(
    input_dir: str,
    output_dir: str,
    features: str = "A,B,C,D,E",
    splits: str = "train.jsonl,val.jsonl,test.jsonl",
) -> None:
    """Apply the requested feature set to each split and write to output_dir."""
    feats = [f.strip().upper() for f in features.split(",") if f.strip()]
    for f in feats:
        if f not in FEATURE_FNS:
            raise ValueError(f"Unknown feature '{f}' (valid: {list(FEATURE_FNS)})")

    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[enrich_fever] input_dir  = {inp}", flush=True)
    print(f"[enrich_fever] output_dir = {out}", flush=True)
    print(f"[enrich_fever] features   = {feats}  "
          f"({[FEATURE_NAMES[f] for f in feats]})", flush=True)

    split_list = [s.strip() for s in splits.split(",")]
    for sp in split_list:
        src = inp / sp
        if not src.exists():
            print(f"  [skip] {src} not found", flush=True)
            continue
        print(f"\n[{sp}] loading {src}", flush=True)
        examples = load_split(src)
        print(f"  {len(examples)} examples loaded", flush=True)
        for f in feats:
            print(f"  -- applying feature {f} ({FEATURE_NAMES[f]}) --", flush=True)
            examples = FEATURE_FNS[f](examples)
        save_split(out / sp, examples)
        print(f"  wrote {out/sp}", flush=True)

    # Copy non-jsonl side files so the enriched dir is a drop-in replacement
    for side in ["dataset.json", "label_map.json"]:
        src = inp / side
        if src.exists():
            shutil.copy2(src, out / side)

    marker = out / ".enriched_features.json"
    marker.write_text(json.dumps({
        "source": str(inp),
        "features": feats,
        "feature_names": [FEATURE_NAMES[f] for f in feats],
    }, indent=2))
    print(f"\n[enrich_fever] wrote marker {marker}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich FEVER with optional semantic/NLI features")
    ap.add_argument("--input_dir", default="data/processed/fever")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--features", required=True,
                    help='Comma-separated subset of A,B,C,D,E (e.g. "A,B" or "B")')
    ap.add_argument("--splits", default="train.jsonl,val.jsonl,test.jsonl")
    args = ap.parse_args()
    enrich_fever(args.input_dir, args.output_dir, args.features, args.splits)


if __name__ == "__main__":
    main()
