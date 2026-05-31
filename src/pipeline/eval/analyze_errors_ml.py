#!/usr/bin/env python3
"""EvoPool: multi-label submodular error clustering -> improvement_brief.json.

Multi-label variant of analyze_errors.py. Same six analyses (pool overview,
per-annotator diagnostics, coverage gap, fragility, error analysis, improvement
targets) but operating on multi-label gold (`true_labels: List[int]`).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # type: ignore[misc]
        return it


ABSTAIN = -1

# Defaults overridden at runtime via --task_name
LABEL_NAMES: Dict[int, str] = {0: "ham", 1: "spam"}
BOOL_FEATURES: List[str] = [
    "has_url", "has_subscribe", "has_channel",
    "has_check_out", "has_email", "has_like",
]

# Tokenization for similarity/keywords (interpretable, rule-friendly)
TOKEN_RE = re.compile(r"[a-z']+")
STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "it", "in", "to", "of", "and", "for",
    "on", "you", "that", "this", "with", "are", "was", "be", "at",
    "or", "not", "but", "from", "have", "has", "had", "do", "does",
    "my", "me", "he", "she", "we", "so", "if", "i", "am", "its",
    "can", "will", "just", "been", "about", "all", "them", "than",
    "would", "what", "when", "who", "which", "how", "no", "there",
    "their", "your", "they", "his", "her", "him", "our", "out",
    "up", "one", "don't", "like", "more", "very", "as", "by",
}

# Text similarity method: "jaccard" (default) or "tfidf"
TEXT_SIM_METHOD: str = "jaccard"


# -----------------------------------------------------------------------
# Embedding-based similarity (precomputed .npz)
# -----------------------------------------------------------------------

class EmbeddingIndex:
    """Loads precomputed L2-normalized embeddings and provides cosine similarity."""

    def __init__(self, npz_path: str, examples: Sequence[Dict[str, Any]]):
        """Load embeddings from .npz and align to example indices by ID."""
        import numpy as _np
        data = _np.load(npz_path, allow_pickle=True)
        emb_matrix = data["embeddings"]  # (N_file, D)
        emb_ids = list(data["ids"])       # (N_file,)
        # Build id -> embedding row mapping
        id_to_row = {str(eid): r for r, eid in enumerate(emb_ids)}
        # Map example indices to embedding vectors
        self._vectors: Dict[int, Any] = {}
        matched = 0
        for idx, ex in enumerate(examples):
            eid = str(ex.get("id", idx))
            if eid in id_to_row:
                self._vectors[idx] = emb_matrix[id_to_row[eid]]
                matched += 1
        print(f"  EmbeddingIndex: matched {matched}/{len(examples)} examples from {npz_path}")

    def cosine(self, i: int, j: int) -> float:
        if i == j:
            return 1.0
        vi = self._vectors.get(i)
        vj = self._vectors.get(j)
        if vi is None or vj is None:
            return 0.0
        # Vectors are L2-normalized, so dot product = cosine similarity
        return float(vi @ vj)


# -----------------------------------------------------------------------
# TF-IDF similarity (no external dependencies)
# -----------------------------------------------------------------------

class TfidfIndex:
    """Lightweight TF-IDF index using only stdlib. Computes cosine similarity."""

    def __init__(self, doc_tokens: Dict[int, List[str]]):
        """Build IDF from a dict of {doc_idx: token_list}."""
        self._doc_ids = sorted(doc_tokens.keys())
        n_docs = len(self._doc_ids)
        # document frequency
        df: Dict[str, int] = Counter()
        for tokens in doc_tokens.values():
            df.update(set(tokens))
        # IDF: log(N / df) with add-1 smoothing
        self._idf: Dict[str, float] = {}
        for term, freq in df.items():
            self._idf[term] = math.log((n_docs + 1) / (freq + 1)) + 1.0
        # Build per-doc TF-IDF vectors (sparse: dict) and norms
        self._vectors: Dict[int, Dict[str, float]] = {}
        self._norms: Dict[int, float] = {}
        for idx in self._doc_ids:
            tokens = doc_tokens[idx]
            tf: Dict[str, int] = Counter(tokens)
            vec: Dict[str, float] = {}
            norm_sq = 0.0
            for term, count in tf.items():
                w = (1 + math.log(count)) * self._idf.get(term, 1.0)
                vec[term] = w
                norm_sq += w * w
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            # normalize
            for term in vec:
                vec[term] /= norm
            self._vectors[idx] = vec
            self._norms[idx] = norm

    def cosine(self, i: int, j: int) -> float:
        """Cosine similarity between two documents (already L2-normalized)."""
        if i == j:
            return 1.0
        va = self._vectors.get(i, {})
        vb = self._vectors.get(j, {})
        if not va or not vb:
            return 0.0
        # dot product over shared terms
        smaller, larger = (va, vb) if len(va) <= len(vb) else (vb, va)
        dot = sum(v * larger[k] for k, v in smaller.items() if k in larger)
        return max(0.0, min(1.0, dot))

# -----------------------------------------------------------------------
# LF-coverage-aware similarity
# -----------------------------------------------------------------------

class LFCoverageIndex:
    """Cosine similarity based on LF firing patterns (coverage vectors).

    Each example is represented by a binary vector of which LFs fire on it.
    Examples covered by the same LFs are more similar.  This signal is weak
    when the pool is small (few LFs) and strong when the pool is large.

    The ``adaptive_alpha`` method returns a blending weight that decreases
    the text-similarity contribution as LF coverage grows, so the
    submodular objective automatically shifts from text-based diversity
    (early iterations) to coverage-aware diversity (later iterations).
    """

    def __init__(self, vote_matrix: List[List[int]], n_examples: int):
        """Build from the existing vote_matrix[lf_idx][example_idx]."""
        import numpy as _np
        n_lfs = len(vote_matrix)
        mat = _np.zeros((n_examples, n_lfs), dtype=_np.float32)
        for j in range(n_lfs):
            for i in range(n_examples):
                if vote_matrix[j][i] != ABSTAIN:
                    mat[i, j] = 1.0
        norms = _np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._normed = mat / norms
        # Coverage fraction (for adaptive alpha)
        self._coverage = float((mat.sum(axis=1) > 0).sum()) / n_examples
        self._n_lfs = n_lfs

    def cosine(self, i: int, j: int) -> float:
        if i == j:
            return 1.0
        return float(self._normed[i] @ self._normed[j])

    def adaptive_alpha(self) -> float:
        """Return text-similarity weight alpha in [0.3, 1.0].

        alpha = 1.0  when coverage < 30%  (pool too immature, trust text sim)
        alpha = 0.3  when coverage > 60%  (mature pool, trust LF coverage)
        Linear interpolation in between.
        """
        cov = self._coverage
        if cov <= 0.30:
            return 1.0
        if cov >= 0.60:
            return 0.3
        # Linear: alpha = 1.0 - 0.7 * (cov - 0.30) / 0.30
        return round(1.0 - 0.7 * (cov - 0.30) / 0.30, 3)


# -----------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


# -----------------------------------------------------------------------
# Load pool & run LFs
# -----------------------------------------------------------------------

def load_pool_module(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pool_module", str(module_path.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def get_annotators(mod: Any) -> List[Tuple[str, Callable]]:
    annotators = getattr(mod, "ANNOTATORS", None)
    if annotators is None:
        raise AttributeError("Pool module has no ANNOTATORS list.")
    return [(fn.__name__, fn) for fn in annotators]


def run_pool(
    annotators: List[Tuple[str, Callable]],
    examples: Sequence[Dict[str, Any]],
) -> List[List[int]]:
    """vote_matrix[lf_idx][ex_idx]"""
    matrix: List[List[int]] = []
    for _, lf in annotators:
        row: List[int] = []
        for ex in examples:
            try:
                v = lf(ex)
            except Exception:
                v = ABSTAIN
            if not isinstance(v, int) or (v != ABSTAIN and v < 0):
                v = ABSTAIN
            row.append(v)
        matrix.append(row)
    return matrix


# -----------------------------------------------------------------------
# Derived per-example helpers
# -----------------------------------------------------------------------

def _example_vote_vector(vote_matrix: List[List[int]], ex_idx: int) -> List[int]:
    return [vote_matrix[j][ex_idx] for j in range(len(vote_matrix))]


def _active_set(votes_for_lf: List[int]) -> Set[int]:
    """Indices where the LF does NOT abstain."""
    return {i for i, v in enumerate(votes_for_lf) if v != ABSTAIN}


def _text_length_bucket(ex: Dict[str, Any]) -> str:
    nw = (ex.get("metadata") or {}).get("num_words", 0)
    if nw < 6:
        return "very_short"
    if nw < 15:
        return "short"
    if nw < 30:
        return "medium"
    return "long"


def _get_text_for_similarity(ex: Dict[str, Any]) -> str:
    meta = ex.get("metadata") or {}
    return str(meta.get("text_normalized") or ex.get("text") or "")


def _token_set(text: str) -> Set[str]:
    return {w for w in TOKEN_RE.findall(text.lower()) if len(w) >= 2 and w not in STOPWORDS}


def _meta_signature(ex: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Discrete, interpretable features that LFs can realistically use.
    Kept small to avoid overfitting and preserve interpretability.
    """
    meta = ex.get("metadata") or {}
    excl = int(meta.get("num_exclamation", 0) or 0)
    excl_bucket = 0 if excl == 0 else 1 if excl == 1 else 2 if excl == 2 else 3
    sig: List[Any] = [bool(meta.get(f, False)) for f in BOOL_FEATURES]
    sig.append(_text_length_bucket(ex))
    sig.append(excl_bucket)
    return tuple(sig)


def _sim_meta(sig_a: Tuple[Any, ...], sig_b: Tuple[Any, ...]) -> float:
    if not sig_a:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def _sim_text_jaccard(tok_a: Set[str], tok_b: Set[str]) -> float:
    if not tok_a and not tok_b:
        return 0.0
    union = tok_a | tok_b
    if not union:
        return 0.0
    return len(tok_a & tok_b) / len(union)


def _sim_examples(
    i: int,
    j: int,
    *,
    tokens_by_idx: Dict[int, Set[str]],
    meta_by_idx: Dict[int, Tuple[Any, ...]],
    sim_beta: float,
    cache: Dict[Tuple[int, int], float],
    tfidf_index: Optional[TfidfIndex] = None,
    embedding_index: Optional[EmbeddingIndex] = None,
    lf_coverage_index: Optional[LFCoverageIndex] = None,
    lf_alpha: Optional[float] = None,
) -> float:
    if i == j:
        return 1.0
    a, b = (i, j) if i < j else (j, i)
    key = (a, b)
    if key in cache:
        return cache[key]
    s_meta = _sim_meta(meta_by_idx[i], meta_by_idx[j])
    if embedding_index is not None:
        s_text = embedding_index.cosine(i, j)
    elif tfidf_index is not None:
        s_text = tfidf_index.cosine(i, j)
    else:
        s_text = _sim_text_jaccard(tokens_by_idx[i], tokens_by_idx[j])
    # Blend text similarity with LF coverage similarity
    if lf_coverage_index is not None and lf_alpha is not None:
        s_lf = lf_coverage_index.cosine(i, j)
        s_text = lf_alpha * s_text + (1.0 - lf_alpha) * s_lf
    s = sim_beta * s_meta + (1.0 - sim_beta) * s_text
    cache[key] = s
    return s


def _build_sim_matrix_numpy(
    universe: List[int],
    candidates: List[int],
    examples: Sequence[Dict[str, Any]],
    sim_beta: float,
    text_sim_method: str,
    embedding_index: Optional[EmbeddingIndex],
    lf_coverage_index: Optional[LFCoverageIndex],
    effective_lf_alpha: Optional[float],
) -> "numpy.ndarray":
    """Precompute sim[i_idx, c_idx] matrix using vectorized numpy ops.

    Returns shape (len(universe), len(candidates)) float32 array.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_indices = sorted(set(universe) | set(candidates))
    idx2pos = {idx: pos for pos, idx in enumerate(all_indices)}

    # --- Text similarity component (sparse TF-IDF or Jaccard) ---
    if text_sim_method in ("tfidf", "hybrid"):
        # Use sklearn TfidfVectorizer for fast sparse TF-IDF + cosine
        texts = [_get_text_for_similarity(examples[idx]) for idx in all_indices]
        vectorizer = TfidfVectorizer(
            max_features=10000, sublinear_tf=True, min_df=2,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        # Compute text sim submatrix: universe × candidates
        u_pos = [idx2pos[i] for i in universe]
        c_pos = [idx2pos[j] for j in candidates]
        text_sim = (tfidf_matrix[u_pos] @ tfidf_matrix[c_pos].T).toarray()
        np.clip(text_sim, 0.0, 1.0, out=text_sim)
    elif text_sim_method == "jaccard":
        # Fallback: vectorized Jaccard (slower but still faster than Python loop)
        token_sets = {idx: _token_set(_get_text_for_similarity(examples[idx])) for idx in all_indices}
        u_pos = list(range(len(universe)))
        c_pos = list(range(len(candidates)))
        text_sim = np.zeros((len(universe), len(candidates)), dtype=np.float32)
        for ui, uidx in enumerate(universe):
            ta = token_sets[uidx]
            if not ta:
                continue
            for ci, cidx in enumerate(candidates):
                tb = token_sets[cidx]
                if not tb:
                    continue
                inter = len(ta & tb)
                union = len(ta | tb)
                text_sim[ui, ci] = inter / union if union > 0 else 0.0
    else:
        text_sim = np.zeros((len(universe), len(candidates)), dtype=np.float32)

    # --- LF coverage similarity component ---
    if lf_coverage_index is not None and effective_lf_alpha is not None and effective_lf_alpha < 1.0:
        u_pos_lf = [universe[i] for i in range(len(universe))]
        c_pos_lf = [candidates[j] for j in range(len(candidates))]
        lf_normed = lf_coverage_index._normed
        lf_sim = lf_normed[u_pos_lf] @ lf_normed[c_pos_lf].T
        np.clip(lf_sim, 0.0, 1.0, out=lf_sim)
        text_sim = effective_lf_alpha * text_sim + (1.0 - effective_lf_alpha) * lf_sim

    # --- Metadata similarity component (vectorized, per-feature) ---
    if sim_beta > 0.0:
        meta_sigs = {idx: _meta_signature(examples[idx]) for idx in all_indices}
        sig_len = len(next(iter(meta_sigs.values()))) if meta_sigs else 1
        if sig_len > 0:
            # Encode mixed-type signatures to int (bools, strings, ints → int codes)
            # Build a per-feature value→code mapping
            all_sigs = [meta_sigs[idx] for idx in all_indices]
            feat_maps: List[Dict] = [{} for _ in range(sig_len)]
            for sig in all_sigs:
                for f, v in enumerate(sig):
                    if v not in feat_maps[f]:
                        feat_maps[f][v] = len(feat_maps[f])
            def _encode(sig):
                return [feat_maps[f][v] for f, v in enumerate(sig)]
            u_meta = np.array([_encode(meta_sigs[i]) for i in universe], dtype=np.int32)
            c_meta = np.array([_encode(meta_sigs[j]) for j in candidates], dtype=np.int32)
            # Accumulate per-feature matches to avoid (n_u, n_c, sig_len) broadcast
            meta_sim = np.zeros((len(universe), len(candidates)), dtype=np.float32)
            for f in range(sig_len):
                meta_sim += np.equal.outer(u_meta[:, f], c_meta[:, f]).astype(np.float32)
            meta_sim /= sig_len
            sim_matrix = sim_beta * meta_sim + (1.0 - sim_beta) * text_sim
        else:
            sim_matrix = text_sim
    else:
        sim_matrix = text_sim

    # Diagonal (self-similarity) = 1.0 where universe and candidate overlap
    u_set = {idx: ui for ui, idx in enumerate(universe)}
    for ci, cidx in enumerate(candidates):
        if cidx in u_set:
            sim_matrix[u_set[cidx], ci] = 1.0

    return sim_matrix.astype(np.float32)


def facility_location_greedy(
    indices: Sequence[int],
    examples: Sequence[Dict[str, Any]],
    *,
    candidate_indices: Optional[Sequence[int]] = None,
    k_max: int,
    sim_beta: float = 0.5,
    k_min: int = 1,
    rho_target: Optional[float] = None,
    weights: Optional[Dict[int, float]] = None,
    seed: int = 42,
    text_sim_method: str = "jaccard",
    embedding_index: Optional[EmbeddingIndex] = None,
    lf_coverage_index: Optional[LFCoverageIndex] = None,
    lf_alpha: Optional[float] = None,
    class_balance_floor: int = 0,
    n_classes: int = 0,
    pseudo_label_for_balance: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Greedy maximization of weighted facility location:
      f(S) = sum_{i in U} w_i * max_{j in S} sim(i, j)

    Uses precomputed numpy similarity matrix + lazy greedy evaluation
    (Minoux 1978) for orders-of-magnitude speedup over naive Python loops.
    """
    import numpy as np

    universe = list(indices)
    if candidate_indices is None:
        candidates = universe[:]
    else:
        candidates = list(candidate_indices)

    if not universe or k_max <= 0 or not candidates:
        return {
            "strategy": "facility_location_greedy",
            "k_max": int(k_max),
            "k_min": int(k_min),
            "rho_target": rho_target,
            "sim_beta": float(sim_beta),
            "selected_indices": [],
            "selected_ids": [],
            "achieved_rho": 0.0,
            "marginal_gains": [],
            "rho_curve": [],
            "W_total": 0.0,
        }

    sim_beta = float(max(0.0, min(1.0, sim_beta)))
    k_min = int(max(1, k_min))
    k_max = int(min(k_max, len(candidates)))
    if rho_target is not None and rho_target <= 0:
        rho_target = None

    rng = random.Random(seed)
    rng.shuffle(candidates)

    w = weights or {}
    weights_list = np.array([float(w.get(i, 1.0)) for i in universe], dtype=np.float32)
    W_total = float(weights_list.sum())
    if W_total <= 0:
        weights_list = np.ones(len(universe), dtype=np.float32)
        W_total = float(len(universe))

    # For hybrid mode, compute adaptive alpha
    effective_lf_alpha = lf_alpha
    if text_sim_method == "hybrid" and lf_coverage_index is not None:
        if effective_lf_alpha is None:
            effective_lf_alpha = lf_coverage_index.adaptive_alpha()
        print(f"  [hybrid] LF coverage={lf_coverage_index._coverage:.1%}, "
              f"alpha={effective_lf_alpha:.3f} "
              f"({effective_lf_alpha:.0%} text + {1-effective_lf_alpha:.0%} LF coverage)")

    # --- Precompute similarity matrix (vectorized) ---
    import time as _time
    t0 = _time.time()
    print(f"  Precomputing similarity matrix ({len(universe)}×{len(candidates)}) ...")
    sim_matrix = _build_sim_matrix_numpy(
        universe, candidates, examples,
        sim_beta=sim_beta,
        text_sim_method=text_sim_method,
        embedding_index=embedding_index,
        lf_coverage_index=lf_coverage_index,
        effective_lf_alpha=effective_lf_alpha,
    )
    print(f"  Similarity matrix built in {_time.time()-t0:.1f}s")

    # --- Lazy greedy selection (Minoux 1978) ---
    # cand_idx_map: position in candidates list -> column in sim_matrix
    n_u = len(universe)
    n_c = len(candidates)

    best_sim_vec = np.zeros(n_u, dtype=np.float32)  # max sim to selected set
    selected: List[int] = []
    selected_ci: Set[int] = set()  # indices into candidates list
    marginal_gains: List[float] = []
    rho_curve: List[float] = []
    f_value = 0.0

    # Upper bounds on marginal gain for lazy evaluation
    upper_bounds = np.full(n_c, np.inf, dtype=np.float32)
    # Track which candidates are still available
    available = np.ones(n_c, dtype=bool)

    # ── Class-balance floor (optional) ──────────────────────────────────
    # Reserve `class_balance_floor` slots per class up-front. For each class
    # we pick the candidate with highest IMPORTANCE WEIGHT among that class's
    # examples. Then standard greedy proceeds for the remaining slots, so the
    # facility-location objective is still maximised over what's left.
    if class_balance_floor and class_balance_floor > 0 and n_classes > 0:
        if (pseudo_label_for_balance is not None
                and len(pseudo_label_for_balance) == len(examples)):
            pseudo_full = list(pseudo_label_for_balance)
        else:
            pseudo_full = [None] * len(examples)

        # Build {class -> ordered candidate-indices (by weight desc)}.
        # cand index ci → universe row when candidate is also in universe;
        # importance weight for class ranking comes from `weights_list[ui]`
        # using ui = universe.index(candidates[ci]) (linear scan; small k).
        u_pos = {u: i for i, u in enumerate(universe)}

        per_class_picked = {c: 0 for c in range(int(n_classes))}
        cand_by_class: Dict[int, List[Tuple[float, int]]] = {c: [] for c in range(int(n_classes))}
        for ci, ex_idx in enumerate(candidates):
            cls = pseudo_full[ex_idx] if ex_idx < len(pseudo_full) else None
            if cls is None or cls < 0 or cls >= int(n_classes):
                continue
            wgt = float(weights_list[u_pos[ex_idx]]) if ex_idx in u_pos else 0.0
            cand_by_class[int(cls)].append((wgt, ci))
        for cls, lst in cand_by_class.items():
            lst.sort(reverse=True)

        # Round-robin pick floor per class — interleaving classes preserves
        # facility-location coverage better than picking all of class 0 first.
        for round_idx in range(int(class_balance_floor)):
            for cls in range(int(n_classes)):
                if per_class_picked[cls] >= int(class_balance_floor):
                    continue
                picked = None
                while cand_by_class[cls]:
                    _w, ci = cand_by_class[cls].pop(0)
                    if available[ci]:
                        picked = ci
                        break
                if picked is None:
                    continue
                ci = picked
                # Add this candidate to the selected set + update FL state.
                selected.append(candidates[ci])
                selected_ci.add(ci)
                available[ci] = False
                sim_col = sim_matrix[:, ci]
                improved = sim_col > best_sim_vec
                gain_now = float(np.dot(weights_list[improved], sim_col[improved] - best_sim_vec[improved]))
                marginal_gains.append(gain_now)
                f_value += gain_now
                np.maximum(best_sim_vec, sim_col, out=best_sim_vec)
                rho_curve.append(float(f_value / W_total) if W_total > 0 else 0.0)
                per_class_picked[cls] += 1
                if len(selected) >= k_max:
                    break
            if len(selected) >= k_max:
                break

    for step in tqdm(range(len(selected), k_max), desc="  Greedy select", leave=False):
        best_candidate_ci: Optional[int] = None
        best_gain = -1.0

        # Lazy greedy: maintain a priority queue via sorted upper bounds
        # Re-evaluate candidates in order of their upper bound; stop when
        # the top candidate's actual gain matches its position
        order = np.argsort(-upper_bounds)  # descending
        for ci in order:
            if not available[ci]:
                continue
            if upper_bounds[ci] <= best_gain:
                # All remaining have lower upper bounds — we're done
                break
            # Compute actual marginal gain
            sim_col = sim_matrix[:, ci]
            delta = sim_col - best_sim_vec
            np.maximum(delta, 0.0, out=delta)
            gain = float(np.dot(weights_list, delta))
            upper_bounds[ci] = gain  # tighten bound
            if gain > best_gain:
                best_gain = gain
                best_candidate_ci = int(ci)

        if best_candidate_ci is None or best_gain <= 0:
            break

        selected.append(candidates[best_candidate_ci])
        selected_ci.add(best_candidate_ci)
        available[best_candidate_ci] = False
        marginal_gains.append(float(best_gain))

        # Update best_sim_vec
        sim_col = sim_matrix[:, best_candidate_ci]
        improved = sim_col > best_sim_vec
        f_value += float(np.dot(weights_list[improved], sim_col[improved] - best_sim_vec[improved]))
        np.maximum(best_sim_vec, sim_col, out=best_sim_vec)

        achieved_rho = (f_value / W_total) if W_total > 0 else 0.0
        rho_curve.append(float(achieved_rho))
        if rho_target is not None and len(selected) >= k_min and achieved_rho >= rho_target:
            break

    achieved_rho = (f_value / W_total) if W_total > 0 else 0.0
    return {
        "strategy": "facility_location_greedy",
        "text_sim_method": text_sim_method,
        "k_max": int(k_max),
        "k_min": int(k_min),
        "rho_target": rho_target,
        "sim_beta": float(sim_beta),
        "selected_indices": selected,
        "selected_ids": [examples[i].get("id") for i in selected],
        "achieved_rho": round(achieved_rho, 4),
        "marginal_gains": [round(g, 6) for g in marginal_gains],
        "rho_curve": [round(r, 6) for r in rho_curve],
        "W_total": round(W_total, 6),
        "class_balance_floor": int(class_balance_floor) if class_balance_floor else 0,
    }


# -----------------------------------------------------------------------
# Selection metrics (paper-aligned): vote stats, weights, submodular selection
# -----------------------------------------------------------------------

def _majority_vote(votes: Sequence[int]) -> int:
    active = [v for v in votes if v != ABSTAIN]
    if not active:
        return ABSTAIN
    c = Counter(active)
    top = c.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return ABSTAIN
    return int(top[0][0])


def _entropy_normalized(probs: Dict[int, float]) -> float:
    """Normalized entropy over a class distribution. Maps to [0, 1]."""
    n_classes = len(probs)
    if n_classes <= 1:
        return 0.0
    h = 0.0
    for p in probs.values():
        if p > 0:
            h -= p * math.log(p)
    max_h = math.log(n_classes)
    return float(h / max_h) if max_h > 0 else 0.0


def compute_vote_stats(vote_matrix: List[List[int]]) -> Dict[str, List[Any]]:
    """
    Computes per-example vote stats based only on LF outputs.
    Returns lists of length n_examples.
    """
    n_lfs = len(vote_matrix)
    n = len(vote_matrix[0]) if n_lfs else 0

    depths: List[int] = []
    agg_labels: List[int] = []
    disagree: List[bool] = []
    vote_entropy: List[float] = []
    vote_probs: List[Dict[int, float]] = []  # per-class probabilities over active votes
    uncertainty_u: List[float] = []

    for i in range(n):
        active = [vote_matrix[j][i] for j in range(n_lfs) if vote_matrix[j][i] != ABSTAIN]
        d = len(active)
        depths.append(d)

        if d == 0:
            agg_labels.append(ABSTAIN)
            disagree.append(False)
            vote_entropy.append(0.0)
            vote_probs.append({})
            uncertainty_u.append(1.0)  # uncovered → max uncertainty
            continue

        counts = Counter(active)
        d_disagree = (len(counts) > 1)
        disagree.append(d_disagree)

        probs = {c: cnt / d for c, cnt in counts.items()}
        vote_probs.append(probs)

        if d_disagree:
            ent = _entropy_normalized(probs)
            vote_entropy.append(ent)
            uncertainty_u.append(ent)
        else:
            vote_entropy.append(0.0)
            uncertainty_u.append(1.0 / (d + 1.0))  # well-supported → low uncertainty

        agg_labels.append(_majority_vote(active))

    return {
        "vote_depth": depths,
        "agg_label": agg_labels,
        "disagree": disagree,
        "vote_entropy": vote_entropy,
        "vote_probs": vote_probs,
        "uncertainty_u": uncertainty_u,
    }


def _quantiles(values: List[float], qs: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {f"q{int(q*100)}": 0.0 for q in qs}
    xs = sorted(values)
    n = len(xs)
    out: Dict[str, float] = {}
    for q in qs:
        q = max(0.0, min(1.0, float(q)))
        idx = int(round(q * (n - 1)))
        out[f"q{int(q*100)}"] = float(xs[idx])
    return out


def sample_seed_indices(
    examples: Sequence[Dict[str, Any]],
    *,
    label_key: str,
    seed: int,
    seed_size: int,
    seed_fraction: float,
) -> List[int]:
    labeled = [i for i, ex in enumerate(examples) if ex.get(label_key) is not None]
    if not labeled:
        return []
    rng = random.Random(seed)
    if seed_size <= 0:
        seed_size = int(round(max(0.0, min(1.0, seed_fraction)) * len(labeled)))
    seed_size = max(1, min(seed_size, len(labeled)))
    return rng.sample(labeled, seed_size)


def estimate_seed_class_stats(
    seed_indices: Sequence[int],
    examples: Sequence[Dict[str, Any]],
    *,
    label_key: str,
    vote_depth: Sequence[int],
) -> Dict[str, Any]:
    """
    Estimates class prior pi_c and class-wise coverage recall R_c from the labeled seed set.
    This matches Eq. (recall-est) in the paper (coverage, not classification recall).
    """
    if not seed_indices:
        return {
            "seed_size": 0,
            "pi_hat": {},
            "recall_hat": {},
        }

    # Guard: if vote_depth is shorter than the example index range (e.g. empty
    # pool with 0 annotators), treat every example as having 0 vote depth
    # rather than raising IndexError. Caller will get an empty pi_hat/recall_hat
    # rather than a cryptic crash.
    vd_len = len(vote_depth) if vote_depth is not None else 0

    class_counts: Counter = Counter()
    class_covered: Counter = Counter()
    for i in seed_indices:
        y = examples[i].get(label_key)
        if y is None:
            continue
        y = int(y)
        class_counts[y] += 1
        if i < vd_len and int(vote_depth[i]) > 0:
            class_covered[y] += 1

    seed_size = sum(class_counts.values())
    pi_hat = {int(c): (class_counts[c] / seed_size) for c in class_counts} if seed_size else {}
    recall_hat = {
        int(c): (class_covered[c] / class_counts[c]) if class_counts[c] else 0.0
        for c in class_counts
    }
    return {
        "seed_size": int(seed_size),
        "pi_hat": {LABEL_NAMES.get(k, str(k)): round(v, 6) for k, v in sorted(pi_hat.items())},
        "pi_hat_by_label": {int(k): float(v) for k, v in pi_hat.items()},
        "recall_hat": {LABEL_NAMES.get(k, str(k)): round(v, 6) for k, v in sorted(recall_hat.items())},
        "recall_hat_by_label": {int(k): float(v) for k, v in recall_hat.items()},
    }


def compute_class_priority_weights(
    recall_hat_by_label: Dict[int, float],
    *,
    epsilon: float = 0.01,
    gamma: float = 1.0,
) -> Dict[int, float]:
    """
    alpha_c ∝ (1 / max(R_c, epsilon))^gamma, normalized to sum to 1.
    """
    epsilon = float(max(1e-9, epsilon))
    gamma = float(max(0.0, gamma))

    labels = sorted(set(LABEL_NAMES.keys()) | set(recall_hat_by_label.keys()))
    raw: Dict[int, float] = {}
    for c in labels:
        r = float(recall_hat_by_label.get(c, 0.0))
        raw[c] = (1.0 / max(r, epsilon)) ** gamma
    Z = sum(raw.values()) or 1.0
    return {c: (raw[c] / Z) for c in raw}


def compute_importance_weights(
    vote_stats: Dict[str, List[Any]],
    *,
    alpha_by_label: Dict[int, float],
    pi_hat_by_label: Dict[int, float],
) -> Dict[str, Any]:
    """
    w_i = u_i * alpha_i, where u_i is uncertainty (Eq. unc) and alpha_i uses
    estimated class priority (Eq. alpha-per-example).
    """
    vote_depth: List[int] = vote_stats["vote_depth"]
    agg_label: List[int] = vote_stats["agg_label"]
    vote_probs: List[Dict[int, float]] = vote_stats["vote_probs"]
    uncertainty_u: List[float] = vote_stats["uncertainty_u"]

    # Expected class weight under prior (used for uncovered)
    expected_alpha_prior = 0.0
    for c, a in alpha_by_label.items():
        expected_alpha_prior += float(pi_hat_by_label.get(c, 0.0)) * float(a)
    if expected_alpha_prior <= 0:
        expected_alpha_prior = sum(alpha_by_label.values()) / max(1, len(alpha_by_label))

    alpha_i: List[float] = []
    w_i: List[float] = []
    for i, d in enumerate(vote_depth):
        if d > 0:
            g = int(agg_label[i])
            if g in alpha_by_label:
                a = float(alpha_by_label[g])
            else:
                # If aggregation abstains, use expected weight under vote distribution
                probs = vote_probs[i]
                a = sum(
                    float(probs.get(c, 0.0)) * float(alpha_by_label.get(c, expected_alpha_prior))
                    for c in alpha_by_label
                )
                if a <= 0:
                    a = expected_alpha_prior
        else:
            a = expected_alpha_prior
        alpha_i.append(a)
        w_i.append(float(uncertainty_u[i]) * float(a))

    return {
        "alpha_i": alpha_i,
        "w_i": w_i,
        "expected_alpha_prior": float(expected_alpha_prior),
    }


def build_facility_clusters(
    universe_indices: Sequence[int],
    center_indices: Sequence[int],
    examples: Sequence[Dict[str, Any]],
    *,
    weights_by_idx: Dict[int, float],
    vote_stats: Dict[str, List[Any]],
    sim_beta: float,
    max_examples_per_cluster: int = 6,
    text_sim_method: str = "jaccard",
    embedding_index: Optional[EmbeddingIndex] = None,
) -> List[Dict[str, Any]]:
    """
    Assign each universe point to its closest selected center (argmax sim),
    producing an interpretable slice decomposition for prompting.
    """
    if not center_indices:
        return []

    universe = list(universe_indices)
    centers = list(center_indices)

    all_indices = sorted(set(universe) | set(centers))
    tokens_by_idx = {i: _token_set(_get_text_for_similarity(examples[i])) for i in all_indices}
    meta_by_idx = {i: _meta_signature(examples[i]) for i in all_indices}
    sim_cache: Dict[Tuple[int, int], float] = {}

    # Build TF-IDF index if requested
    tfidf_index: Optional[TfidfIndex] = None
    if text_sim_method == "tfidf":
        doc_tokens = {i: list(tokens_by_idx[i]) for i in all_indices}
        tfidf_index = TfidfIndex(doc_tokens)

    # assignment: center -> list of member indices
    members: Dict[int, List[int]] = {c: [] for c in centers}
    best_center_sim: Dict[int, float] = {}
    for i in universe:
        best_c = centers[0]
        best_s = -1.0
        for c in centers:
            s = _sim_examples(
                i, c,
                tokens_by_idx=tokens_by_idx,
                meta_by_idx=meta_by_idx,
                sim_beta=sim_beta,
                cache=sim_cache,
                tfidf_index=tfidf_index,
                embedding_index=embedding_index,
            )
            if s > best_s:
                best_s = s
                best_c = c
        members[best_c].append(i)
        best_center_sim[i] = float(best_s)

    # build cluster summaries
    vote_depth: List[int] = vote_stats["vote_depth"]
    disagree: List[bool] = vote_stats["disagree"]
    uncertainty_u: List[float] = vote_stats["uncertainty_u"]
    alpha_i: Optional[List[float]] = vote_stats.get("alpha_i")  # may be added later
    w_i: Optional[List[float]] = vote_stats.get("w_i")  # may be added later
    agg_label: List[int] = vote_stats["agg_label"]

    # Background texts for keywords: use entire universe for contrast
    background_texts = [_get_text_for_similarity(examples[i]) for i in universe]

    clusters: List[Dict[str, Any]] = []
    total_weight_mass = sum(float(weights_by_idx.get(i, 0.0)) for i in universe) or 1.0
    for c in centers:
        idxs = members.get(c, [])
        if not idxs:
            continue
        weight_mass = sum(float(weights_by_idx.get(i, 0.0)) for i in idxs)

        # top examples by weight within cluster
        idxs_sorted = sorted(idxs, key=lambda i: float(weights_by_idx.get(i, 0.0)), reverse=True)
        top_idxs = idxs_sorted[:max_examples_per_cluster]
        rep_examples: List[Dict[str, Any]] = []
        for i in top_idxs:
            ex = examples[i]
            rep_examples.append({
                "id": ex.get("id"),
                "text": str(ex.get("text", ""))[:400],
                "vote_depth": int(vote_depth[i]),
                "agg_label": int(agg_label[i]),
                "agg_label_name": LABEL_NAMES.get(int(agg_label[i]), "abstain") if int(agg_label[i]) != ABSTAIN else "abstain",
                "disagree": bool(disagree[i]),
                "vote_uncertainty_u": round(float(uncertainty_u[i]), 6),
                "alpha_i": round(float(alpha_i[i]), 6) if alpha_i is not None else None,
                "w_i": round(float(w_i[i]), 6) if w_i is not None else None,
                "metadata_profile": _feature_profile(ex),
                "best_center_sim": round(float(best_center_sim.get(i, 0.0)), 6),
            })

        # cluster keywords from top-weight texts (cap for stability)
        target_texts = [_get_text_for_similarity(examples[i]) for i in idxs_sorted[: min(len(idxs_sorted), 250)]]
        keywords = _extract_distinctive_keywords(target_texts, background_texts, top_k=15, min_count=3)

        clusters.append({
            "center_id": examples[c].get("id"),
            "center_index": int(c),
            "center_text": str(examples[c].get("text", ""))[:400],
            "center_profile": _feature_profile(examples[c]),
            "cluster_size": int(len(idxs)),
            "cluster_weight_mass": round(float(weight_mass), 6),
            "cluster_weight_fraction": round(float(weight_mass / total_weight_mass), 6),
            "cluster_uncovered_fraction": round(sum(1 for i in idxs if int(vote_depth[i]) == 0) / len(idxs), 6),
            "cluster_fragile_fraction": round(sum(1 for i in idxs if int(vote_depth[i]) == 1) / len(idxs), 6),
            "cluster_disagree_fraction": round(sum(1 for i in idxs if bool(disagree[i])) / len(idxs), 6),
            "distinctive_keywords": keywords,
            "top_examples": rep_examples,
        })

    clusters.sort(key=lambda c: -c["cluster_weight_mass"])
    return clusters


# -----------------------------------------------------------------------
# BADGE and BatchBALD: principled acquisition alternatives to submodular
# -----------------------------------------------------------------------
#
# Both methods share a "per-annotator predictive distribution" view:
#     P(y | x, theta_j), where each annotator a_j is a posterior sample theta_j.
#
# Given the confirmed TFIDF-prototype hybrid aggregator, we also have the
# aggregated p_bar(y | x).  For selection, we reuse these signals rather than
# retraining a student model between iterations.


def _compute_annotator_pred_probs(
    vote_matrix: List[List[int]],
    n_examples: int,
    n_classes: int,
    *,
    smooth: float = 0.1,
    abstain_prior: Optional["numpy.ndarray"] = None,
) -> "numpy.ndarray":
    """Annotator-conditional predictive distribution P(y | x, theta_j).

    If a_j(x) = c: sharp-with-smoothing distribution — (1 - smooth) on c,
    smooth/(C-1) on each other class.  If a_j abstains on x: use
    ``abstain_prior`` (class prior pi_hat) if provided, else uniform.

    Returns array of shape (N, K, C).
    """
    import numpy as np
    n_lfs = len(vote_matrix)
    if abstain_prior is None:
        abstain_prior = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)
    abstain_prior = np.asarray(abstain_prior, dtype=np.float32)
    if abstain_prior.shape[0] != n_classes:
        abstain_prior = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)
    off = smooth / max(1, n_classes - 1)
    probs = np.empty((n_examples, n_lfs, n_classes), dtype=np.float32)
    for j in range(n_lfs):
        row = vote_matrix[j]
        for i in range(n_examples):
            v = row[i]
            if v == ABSTAIN or v < 0 or v >= n_classes:
                probs[i, j] = abstain_prior
            else:
                probs[i, j].fill(off)
                probs[i, j, int(v)] = 1.0 - smooth
    return probs


def _soft_majority_probs(
    vote_matrix: List[List[int]],
    n_examples: int,
    n_classes: int,
    *,
    laplace: float = 1.0,
    fallback_prior: Optional["numpy.ndarray"] = None,
) -> "numpy.ndarray":
    """Aggregate soft predictive P(y | x) via Laplace-smoothed vote counts.

    Returns array of shape (N, C).  Used by BADGE as the "model prediction".
    """
    import numpy as np
    n_lfs = len(vote_matrix)
    if fallback_prior is None:
        fallback_prior = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)
    counts = np.zeros((n_examples, n_classes), dtype=np.float32)
    for j in range(n_lfs):
        row = vote_matrix[j]
        for i in range(n_examples):
            v = row[i]
            if 0 <= v < n_classes:
                counts[i, int(v)] += 1.0
    totals = counts.sum(axis=1)
    out = np.empty_like(counts)
    uncovered = totals == 0
    out[~uncovered] = (counts[~uncovered] + laplace) / (
        totals[~uncovered, None] + laplace * n_classes
    )
    out[uncovered] = np.asarray(fallback_prior, dtype=np.float32)
    return out


def badge_select(
    examples: Sequence[Dict[str, Any]],
    vote_matrix: List[List[int]],
    *,
    k: int,
    n_classes: int,
    tfidf_dim: int = 128,
    laplace: float = 1.0,
    fallback_prior: Optional["numpy.ndarray"] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """BADGE (Ash et al., ICLR 2020): k-means++ over gradient embeddings.

    Gradient embedding g(x) = (p(y|x) - 1_{y_hat}) ⊗ phi(x), where p(y|x) is
    the soft-majority prediction (Laplace-smoothed) and phi(x) is a TF-IDF
    projection to ``tfidf_dim`` (via TruncatedSVD).

    k-means++ selects k points with probability proportional to squared
    distance from the current set, which approximates A-optimal experimental
    design over the gradient space.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    n = len(examples)
    if k <= 0 or n == 0:
        return {
            "strategy": "badge",
            "selected_indices": [],
            "marginal_gains": [],
            "rho_curve": [],
            "k": 0,
        }
    k = int(min(k, n))

    # 1. TF-IDF features, reduced via truncated SVD to a compact embedding.
    texts = [_get_text_for_similarity(ex) for ex in examples]
    vectorizer = TfidfVectorizer(
        max_features=10000, sublinear_tf=True, min_df=2,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    tfidf = vectorizer.fit_transform(texts)
    d_target = int(min(tfidf_dim, tfidf.shape[1] - 1, max(1, n - 1)))
    if d_target < 1:
        d_target = 1
    svd = TruncatedSVD(n_components=d_target, random_state=seed)
    phi = svd.fit_transform(tfidf).astype(np.float32)  # (N, D)

    # 2. Soft-majority predictive distribution p(y|x).
    p = _soft_majority_probs(
        vote_matrix, n, n_classes,
        laplace=laplace, fallback_prior=fallback_prior,
    )  # (N, C)
    y_hat = p.argmax(axis=1)  # (N,)

    # 3. Gradient embedding:  g(x)[c, :] = (p_c - 1_{c=y_hat}) * phi(x)
    #    Flatten over classes for distance computation.  Shape (N, C*D).
    C = n_classes
    D = phi.shape[1]
    resid = p.copy()
    resid[np.arange(n), y_hat] -= 1.0  # (N, C)
    # Outer product done implicitly by tiling: g[i] = concat_c [resid[i, c] * phi[i]]
    g = (resid[:, :, None] * phi[:, None, :]).reshape(n, C * D).astype(np.float32)

    # 4. k-means++ seeding over g.
    rng = np.random.default_rng(seed)
    # First point: sampled with probability proportional to ||g_i||^2 so that
    # high-uncertainty examples (large gradient norm) are preferred.
    sq_norms = (g * g).sum(axis=1)
    if sq_norms.sum() <= 0:
        first = int(rng.integers(0, n))
    else:
        probs0 = sq_norms / sq_norms.sum()
        first = int(rng.choice(n, p=probs0))
    selected: List[int] = [first]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[first] = True

    # Maintain min squared distance to the selected set.
    diff = g - g[first]
    min_d2 = (diff * diff).sum(axis=1)
    min_d2[first] = 0.0
    gains: List[float] = [float(sq_norms[first])]

    for _ in range(1, k):
        total = float(min_d2.sum())
        if total <= 0:
            # Degenerate: pick any remaining index uniformly.
            remaining = np.where(~selected_mask)[0]
            if remaining.size == 0:
                break
            nxt = int(rng.choice(remaining))
        else:
            probs = min_d2 / total
            probs[selected_mask] = 0.0
            probs = probs / probs.sum() if probs.sum() > 0 else probs
            nxt = int(rng.choice(n, p=probs))
        selected.append(nxt)
        selected_mask[nxt] = True
        gains.append(float(min_d2[nxt]))
        # Update min distance.
        diff = g - g[nxt]
        new_d2 = (diff * diff).sum(axis=1)
        min_d2 = np.minimum(min_d2, new_d2)
        min_d2[nxt] = 0.0

    return {
        "strategy": "badge",
        "k": int(len(selected)),
        "selected_indices": selected,
        "marginal_gains": [round(g_, 6) for g_ in gains],
        "per_example_scores": [round(float(x), 6) for x in sq_norms.tolist()],
        "rho_curve": [],
        "tfidf_dim": int(D),
        "n_classes": int(C),
    }


def batchbald_select(
    examples: Sequence[Dict[str, Any]],
    vote_matrix: List[List[int]],
    *,
    k: int,
    n_classes: int,
    smooth: float = 0.1,
    abstain_prior: Optional["numpy.ndarray"] = None,
    n_mc_samples: int = 1000,
    prefilter_multiplier: int = 8,
    seed: int = 42,
    carryover_age_by_id: Optional[Dict[str, int]] = None,
    carryover_lambda: float = 0.1,
    carryover_age_cap: int = 4,
    class_balance_floor: int = 0,
    pseudo_label_for_balance: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """BatchBALD (Kirsch et al., NeurIPS 2019): greedy MI with MC joint entropy.

    I(Y_S; theta | X_S) = H(Y_S | X_S) - sum_{x in S} E_theta[H(y|x, theta)]

    Each annotator a_j is a theta sample, giving K = n_annotators posterior
    samples.  MC-estimated joint entropy via ancestral sampling of class
    configurations under the predictive distribution.  Greedy selection has
    a (1 - 1/e) approximation guarantee because I(Y_S; theta) is submodular.

    We pre-filter to the top ``prefilter_multiplier * k`` candidates by
    per-example BALD score to keep runtime tractable.
    """
    import numpy as np
    n = len(examples)
    K = len(vote_matrix)
    C = int(n_classes)
    if k <= 0 or n == 0 or K == 0:
        return {
            "strategy": "batchbald",
            "selected_indices": [],
            "marginal_gains": [],
            "rho_curve": [],
            "k": 0,
        }
    k = int(min(k, n))

    # (N, K, C) per-annotator predictive distribution
    pred = _compute_annotator_pred_probs(
        vote_matrix, n, C,
        smooth=smooth,
        abstain_prior=abstain_prior,
    )
    eps = 1e-12
    # p_bar(y | x) = mean over theta samples; predictive entropy H(p_bar).
    p_bar = pred.mean(axis=1)  # (N, C)
    pred_H = -(p_bar * np.log(p_bar + eps)).sum(axis=1)  # (N,)
    # E_theta[H(y | x, theta)] = mean_j H(pred[i, j])
    cond_H_per_j = -(pred * np.log(pred + eps)).sum(axis=2)  # (N, K)
    cond_H = cond_H_per_j.mean(axis=1)                         # (N,)
    bald_raw = pred_H - cond_H                                 # (N,) — original BALD

    # ── Carryover-aware BALD boost ───────────────────────────────────────
    # A point that the selector has flagged high-MI in previous iterations
    # AND that remained uncovered next round is a *persistent* hard point.
    # Boost its BALD score by (1 + λ · carryover_age_i), capped at
    # carryover_age_cap rounds (beyond which we suspect it is out of the
    # LLM's reachable pattern support — Assumption~1 in the paper).
    boost = np.ones(n, dtype=np.float64)
    carryover_boosted = 0
    if carryover_age_by_id:
        for i in range(n):
            rid = examples[i].get("id")
            if rid is None:
                continue
            age = int(carryover_age_by_id.get(str(rid), 0))
            if age <= 0:
                continue
            age = min(age, int(carryover_age_cap))
            boost[i] = 1.0 + float(carryover_lambda) * age
            carryover_boosted += 1
    bald = bald_raw * boost

    # Pre-filter candidates.
    m = int(min(n, max(k, prefilter_multiplier * k)))
    cand = np.argsort(-bald)[:m].tolist()
    cand_set = set(cand)

    # ── Class-balance floor (optional) ──────────────────────────────────
    # If `class_balance_floor > 0`, enforce that each of the C classes gets
    # at least `class_balance_floor` slots in the selected batch (under a
    # pseudo-label assignment derived from the ensemble's predictive mean).
    # The remaining `k - C * class_balance_floor` slots are filled by the
    # standard greedy MI procedure starting from the floor-selected set.
    # This makes BatchBALD class-aware without abandoning the MI objective.
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    gains: List[float] = []

    # Pseudo-labels: use caller-provided (e.g. majority vote / aggregator
    # argmax) when available; otherwise fall back to argmax of p_bar.
    if pseudo_label_for_balance is not None and len(pseudo_label_for_balance) == n:
        pseudo = np.asarray(pseudo_label_for_balance, dtype=np.int64)
    else:
        pseudo = p_bar.argmax(axis=1)  # (N,)

    if class_balance_floor and class_balance_floor > 0:
        floor = int(class_balance_floor)
        # Indices in cand sorted by BALD score, descending.
        cand_sorted = sorted(cand, key=lambda i: -float(bald[i]))
        per_class_picked = {c: 0 for c in range(C)}
        for i in cand_sorted:
            cls = int(pseudo[i])
            if cls < 0 or cls >= C:
                continue
            if per_class_picked[cls] < floor and len(selected) < k:
                selected.append(i)
                gains.append(float(bald[i]))
                per_class_picked[cls] += 1
            if len(selected) >= k:
                break

    # If floor didn't fill anything (e.g. floor=0 or pseudo all -1), seed with
    # the global top-BALD candidate.
    if not selected:
        first = int(max(cand, key=lambda i: float(bald[i])))
        selected.append(first)
        gains.append(float(bald[first]))

    # Maintain log p(y_n | theta_k) for n MC samples under current S.
    log_pred = np.log(pred + eps)  # (N, K, C)

    # Initialise log_p_Sk by drawing MC configs for every already-selected id
    # (could be one — the global top — or many, if class_balance_floor pre-filled).
    log_p_Sk = None
    cond_H_cum = 0.0
    for sel_i in selected:
        configs_i = rng.choice(C, size=n_mc_samples, p=p_bar[sel_i])  # (n_mc,)
        add = log_pred[sel_i][:, configs_i].T  # (n_mc, K)
        log_p_Sk = add if log_p_Sk is None else (log_p_Sk + add)
        cond_H_cum += float(cond_H[sel_i])

    # Current joint entropy estimate H(Y_S) via logsumexp-mean trick.
    def _joint_H(log_p_Sk_local: "numpy.ndarray") -> float:
        max_ = log_p_Sk_local.max(axis=1, keepdims=True)
        log_p = np.log(np.exp(log_p_Sk_local - max_).sum(axis=1) + eps) + max_.squeeze(-1)
        log_p -= np.log(K)
        return float(-log_p.mean())

    H_cur = _joint_H(log_p_Sk)

    # Cache selected set for O(1) membership check.
    selected_set: set = set(selected)

    # Greedy fill the remaining slots using marginal MI.
    for step in range(len(selected), k):
        best_i = -1
        best_gain = -1e18
        best_log_p_new = None
        best_configs_new = None
        # Filter candidates once, then iterate.
        remaining = [i for i in cand if i not in selected_set]
        if not remaining:
            break
        # Vectorized MC sampling: draw n_mc class samples for each candidate at
        # once (shape: (len(remaining), n_mc)) rather than calling rng.choice
        # inside the hot loop.  Each row is drawn under p_bar[i] (different
        # probability vectors), so we must sample per-row — but we can do this
        # as a single numpy op using the inverse-CDF trick.
        p_cand = p_bar[remaining]                     # (R, C)
        cdf = np.cumsum(p_cand, axis=1)               # (R, C)
        cdf = cdf / cdf[:, -1:]                       # normalize (guard)
        u = rng.random(size=(len(remaining), n_mc_samples))  # (R, n_mc)
        # searchsorted per row: compare u vs cdf expanded to (R, n_mc, C)
        # Memory-lite: iterate rows but keep vectorization per-row.
        for ri, i in enumerate(remaining):
            y_samples = np.searchsorted(cdf[ri], u[ri])  # (n_mc,)
            # log_pred[i] shape (K, C); select via y_samples: (K, n_mc) then .T
            add = log_pred[i][:, y_samples].T        # (n_mc, K)
            new_log_p_Sk = log_p_Sk + add
            H_new = _joint_H(new_log_p_Sk)
            gain = H_new - H_cur - float(cond_H[i])
            if gain > best_gain:
                best_gain = gain
                best_i = int(i)
                best_log_p_new = new_log_p_Sk
                best_configs_new = y_samples
        if best_i < 0:
            break
        selected.append(best_i)
        selected_set.add(best_i)
        gains.append(float(best_gain))
        log_p_Sk = best_log_p_new
        H_cur = _joint_H(log_p_Sk)
        cond_H_cum += float(cond_H[best_i])

    return {
        "strategy": "batchbald",
        "k": int(len(selected)),
        "selected_indices": selected,
        "marginal_gains": [round(g_, 6) for g_ in gains],
        "per_example_scores": [round(float(x), 6) for x in bald.tolist()],
        "per_example_scores_raw": [round(float(x), 6) for x in bald_raw.tolist()],
        "rho_curve": [],
        "pre_filter_m": int(m),
        "n_mc_samples": int(n_mc_samples),
        "n_theta_samples_K": int(K),
        "smooth": float(smooth),
        "carryover_boost_applied": {
            "n_boosted": int(carryover_boosted),
            "lambda": float(carryover_lambda),
            "age_cap": int(carryover_age_cap),
        } if carryover_age_by_id else None,
        "class_balance_floor": int(class_balance_floor) if class_balance_floor else 0,
    }


def compute_query_selection_paper_method(
    annotators: List[Tuple[str, Callable]],
    vote_matrix: List[List[int]],
    examples: List[Dict[str, Any]],
    *,
    label_key: str,
    seed: int,
    seed_size: int,
    seed_fraction: float,
    class_weight_epsilon: float,
    class_weight_gamma: float,
    query_k_min: int,
    query_k_max: int,
    query_rho_target: Optional[float],
    query_sim_beta: float,
    text_sim_method: str = "jaccard",
    embeddings_path: Optional[str] = None,
    query_selection_method: str = "submodular",
    max_top_weight_examples: int = 30,
    carryover_age_by_id: Optional[Dict[str, int]] = None,
    carryover_lambda: float = 0.1,
    carryover_age_cap: int = 4,
    class_balance_floor: int = 0,
    a1_val_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Implements the paper-aligned selection:
      - compute vote-derived uncertainty u_i
      - estimate class priority weights alpha_c from labeled seed set
      - compute w_i = u_i * alpha_i using estimated classes (agg label) + priors
      - select a batch via greedy facility location with adaptive stopping
      - produce cluster summaries for prompting
    """
    n = len(examples)
    n_lfs = len(annotators)

    vote_stats = compute_vote_stats(vote_matrix)

    # Seed estimation (requires labels on the seed subset)
    seed_indices = sample_seed_indices(
        examples,
        label_key=label_key,
        seed=seed,
        seed_size=seed_size,
        seed_fraction=seed_fraction,
    )
    seed_stats = estimate_seed_class_stats(
        seed_indices,
        examples,
        label_key=label_key,
        vote_depth=vote_stats["vote_depth"],
    )
    pi_hat_by_label: Dict[int, float] = seed_stats.get("pi_hat_by_label", {}) or {}
    recall_hat_by_label: Dict[int, float] = seed_stats.get("recall_hat_by_label", {}) or {}

    # If no seed labels are available, fall back to uniform priors and equal class weights
    n_cls = len(LABEL_NAMES) or 2
    if not pi_hat_by_label:
        pi_hat_by_label = {c: 1.0 / n_cls for c in LABEL_NAMES} if LABEL_NAMES else {0: 0.5, 1: 0.5}
    if not recall_hat_by_label:
        recall_hat_by_label = {c: 0.0 for c in LABEL_NAMES} if LABEL_NAMES else {0: 0.0, 1: 0.0}

    alpha_by_label = compute_class_priority_weights(
        recall_hat_by_label,
        epsilon=class_weight_epsilon,
        gamma=class_weight_gamma,
    )
    alpha_by_name = {LABEL_NAMES.get(k, str(k)): round(float(v), 6) for k, v in sorted(alpha_by_label.items())}

    imp = compute_importance_weights(
        vote_stats,
        alpha_by_label=alpha_by_label,
        pi_hat_by_label=pi_hat_by_label,
    )
    alpha_i: List[float] = imp["alpha_i"]
    w_i: List[float] = imp["w_i"]

    # Attach for downstream cluster builder (optional fields)
    vote_stats["alpha_i"] = alpha_i
    vote_stats["w_i"] = w_i

    weights_by_idx = {i: float(w_i[i]) for i in range(n)}
    W_total = sum(weights_by_idx.values()) or 1.0

    # Weight mass diagnostics by vote depth category
    depth = vote_stats["vote_depth"]
    disagree = vote_stats["disagree"]
    depth_bucket_mass: Dict[str, float] = {
        "depth0_uncovered": 0.0,
        "depth1_fragile": 0.0,
        "depth2": 0.0,
        "depth3plus": 0.0,
    }
    disagree_mass = 0.0
    for i in range(n):
        wi = float(weights_by_idx[i])
        d = int(depth[i])
        if d == 0:
            depth_bucket_mass["depth0_uncovered"] += wi
        elif d == 1:
            depth_bucket_mass["depth1_fragile"] += wi
        elif d == 2:
            depth_bucket_mass["depth2"] += wi
        else:
            depth_bucket_mass["depth3plus"] += wi
        if bool(disagree[i]):
            disagree_mass += wi

    # Batch selection — method depends on query_selection_method
    universe = list(range(n))
    import random as _random

    # Build embedding index if semantic similarity requested (needed for clustering too)
    emb_index: Optional[EmbeddingIndex] = None
    if text_sim_method == "semantic" and embeddings_path:
        emb_index = EmbeddingIndex(embeddings_path, examples)

    # Build LF coverage index for hybrid similarity
    lf_cov_index: Optional[LFCoverageIndex] = None
    if text_sim_method == "hybrid":
        lf_cov_index = LFCoverageIndex(vote_matrix, n)
        print(f"  LF coverage index: {lf_cov_index._n_lfs} LFs, "
              f"coverage={lf_cov_index._coverage:.1%}, "
              f"adaptive_alpha={lf_cov_index.adaptive_alpha():.3f}")

    if query_selection_method == "uncertainty":
        # Top-k by importance weight w_i (uncertainty × class priority)
        ranked = sorted(range(n), key=lambda i: float(weights_by_idx[i]), reverse=True)
        sel_k = query_k_max
        sel_indices = ranked[:sel_k]
        acq_scores = [float(weights_by_idx[i]) for i in sel_indices]
        selection = {
            "method": "uncertainty_topk",
            "method_label": "uncertainty (importance-weight top-k: u_i * alpha_i)",
            "selection_rationale": "examples ranked by classical uncertainty × class-priority; no diversity constraint",
            "score_name": "importance_weight",
            "acquisition_scores": acq_scores,
            "per_example_scores": [float(weights_by_idx[i]) for i in range(n)],
            "selected_indices": sel_indices,
            "k": sel_k,
            "achieved_rho": None,
            "marginal_gains": acq_scores,
            "rho_curve": [],
        }
    elif query_selection_method == "random":
        # Uniform random sampling
        rng = _random.Random(seed)
        sel_k = query_k_max
        sel_indices = rng.sample(universe, min(sel_k, n))
        acq_scores = [float(weights_by_idx[i]) for i in sel_indices]
        selection = {
            "method": "random",
            "method_label": "uniform random sampling (no acquisition signal)",
            "selection_rationale": "control baseline; examples drawn uniformly at random without any informativeness signal",
            "score_name": "none",
            "acquisition_scores": acq_scores,
            "selected_indices": sel_indices,
            "k": sel_k,
            "achieved_rho": None,
            "marginal_gains": acq_scores,
            "rho_curve": [],
        }
    elif query_selection_method == "badge":
        import numpy as _np
        n_cls_eff = len(LABEL_NAMES) or n_cls
        fallback_prior = _np.array(
            [float(pi_hat_by_label.get(c, 1.0 / n_cls_eff)) for c in range(n_cls_eff)],
            dtype=_np.float32,
        )
        selection = badge_select(
            examples,
            vote_matrix,
            k=query_k_max,
            n_classes=n_cls_eff,
            tfidf_dim=128,
            laplace=1.0,
            fallback_prior=fallback_prior,
            seed=seed,
        )
    elif query_selection_method == "batchbald":
        import numpy as _np
        n_cls_eff = len(LABEL_NAMES) or n_cls
        abstain_prior = _np.array(
            [float(pi_hat_by_label.get(c, 1.0 / n_cls_eff)) for c in range(n_cls_eff)],
            dtype=_np.float32,
        )
        # Pseudo-label per example for class-balance floor (use majority-vote
        # aggregate label; ABSTAIN → -1 → ignored by the floor).
        _pseudo = list(vote_stats["agg_label"])
        selection = batchbald_select(
            examples,
            vote_matrix,
            k=query_k_max,
            n_classes=n_cls_eff,
            smooth=0.1,
            abstain_prior=abstain_prior,
            n_mc_samples=500,
            prefilter_multiplier=8,
            seed=seed,
            carryover_age_by_id=carryover_age_by_id,
            carryover_lambda=carryover_lambda,
            carryover_age_cap=carryover_age_cap,
            class_balance_floor=int(class_balance_floor),
            pseudo_label_for_balance=_pseudo,
        )
    elif query_selection_method in {"a1_margin", "a1_entropy_cb", "a1_v2_kl",
                                      "text_cluster_a1err", "disagree_submod",
                                      "a1_nbrconflict"}:
        # A1-aware selection: delegate to a1_sel_ablation.a1_signals
        import sys as _sys
        from pathlib import Path as _P
        _here = _P(__file__).resolve().parent
        _a1dir = str(_here / "a1_sel_ablation")
        if _a1dir not in _sys.path:
            _sys.path.insert(0, _a1dir)
        from a1_signals import compute_a1_selection  # type: ignore
        # Need val_rows to fit A1; pass through global var bound earlier.
        # `examples` is the universe (train); val labels come from val_rows in
        # the outer caller context. analyze() must have val_rows available.
        if a1_val_rows is None:
            # Fall back to using examples themselves as their own val (will overfit
            # — caller should pass a1_val_rows for honest results).
            print("  [a1-aware] WARNING: a1_val_rows not provided; using train examples "
                  "as val (results will be optimistic).")
            _val_rows = examples
        else:
            _val_rows = a1_val_rows
        selection = compute_a1_selection(
            method=query_selection_method,
            examples=examples,
            vote_matrix=vote_matrix,
            weights_by_idx=weights_by_idx,
            val_rows=_val_rows,
            val_label_key=label_key,
            k=query_k_max,
            n_classes=int(len(LABEL_NAMES)) if LABEL_NAMES else int(n_cls),
            seed=seed,
            class_balance_floor=int(class_balance_floor) if class_balance_floor else 1,
        )
    else:
        # Submodular facility location (default)
        _pseudo = list(vote_stats["agg_label"])
        selection = facility_location_greedy(
            universe,
            examples,
            candidate_indices=None,
            k_max=query_k_max,
            k_min=query_k_min,
            rho_target=query_rho_target,
            sim_beta=query_sim_beta,
            weights=weights_by_idx,
            seed=seed,
            text_sim_method=text_sim_method,
            embedding_index=emb_index,
            lf_coverage_index=lf_cov_index,
            class_balance_floor=int(class_balance_floor),
            n_classes=int(len(LABEL_NAMES)) if LABEL_NAMES else 0,
            pseudo_label_for_balance=_pseudo,
        )

    # Normalize method-identity fields so downstream (Improver/Refiner) knows the
    # selection method, its rationale, and an acquisition score per selected idx.
    _method_meta = {
        "uncertainty_topk": (
            "uncertainty (importance-weight top-k: u_i * alpha_i)",
            "examples ranked by classical uncertainty × class-priority; no diversity constraint",
            "importance_weight",
        ),
        "random": (
            "uniform random sampling (no acquisition signal)",
            "control baseline; examples drawn uniformly at random",
            "none",
        ),
        "badge": (
            "BADGE (k-means++ on gradient embeddings)",
            "gradient-embedding seeds maximizing expected model update magnitude and diversity (Ash et al., 2020)",
            "badge_distance_gain",
        ),
        "batchbald": (
            "BatchBALD (greedy mutual information)",
            "batch mutual information I(Y_S;θ|X_S); selects examples that are jointly informative — high disagreement among annotators AND non-redundant across the batch (Kirsch et al., 2019)",
            "batch_mi_gain",
        ),
        "facility_location": (
            "submodular facility location (weighted coverage)",
            "diversity-maximizing selection covering the uncertainty-weighted universe with an adaptive rho stopping rule",
            "facility_location_gain",
        ),
    }
    _strat_key = selection.get("strategy") or selection.get("method") or query_selection_method
    if _strat_key in _method_meta:
        label, rationale, score_name = _method_meta[_strat_key]
        selection.setdefault("method_label", label)
        selection.setdefault("selection_rationale", rationale)
        selection.setdefault("score_name", score_name)
    if "acquisition_scores" not in selection:
        selection["acquisition_scores"] = [
            float(g) for g in selection.get("marginal_gains", [])
        ]

    selected_indices: List[int] = selection["selected_indices"]

    # Build selected example records (for prompting + reproducibility)
    selected_examples: List[Dict[str, Any]] = []
    for rank, idx in enumerate(selected_indices, start=1):
        ex = examples[idx]
        active_votes = {
            annotators[j][0]: vote_matrix[j][idx]
            for j in range(n_lfs)
            if vote_matrix[j][idx] != ABSTAIN
        }
        agg = int(vote_stats["agg_label"][idx])
        selected_examples.append({
            "rank": int(rank),
            "index": int(idx),
            "id": ex.get("id"),
            "text": str(ex.get("text", ""))[:400],
            "metadata_profile": _feature_profile(ex),
            "vote_depth": int(vote_stats["vote_depth"][idx]),
            "disagree": bool(vote_stats["disagree"][idx]),
            "vote_entropy": round(float(vote_stats["vote_entropy"][idx]), 6),
            "agg_label": agg,
            "agg_label_name": LABEL_NAMES.get(agg, "abstain") if agg != ABSTAIN else "abstain",
            "uncertainty_u": round(float(vote_stats["uncertainty_u"][idx]), 6),
            "alpha_i": round(float(alpha_i[idx]), 6),
            "w_i": round(float(w_i[idx]), 6),
            "active_votes": active_votes,
            "marginal_gain": selection.get("marginal_gains", [])[rank - 1] if rank - 1 < len(selection.get("marginal_gains", [])) else None,
            "rho_after": selection.get("rho_curve", [])[rank - 1] if rank - 1 < len(selection.get("rho_curve", [])) else None,
        })

    # Facility-location induced clusters (slice decomposition)
    clusters = build_facility_clusters(
        universe,
        selected_indices,
        examples,
        weights_by_idx=weights_by_idx,
        vote_stats=vote_stats,
        sim_beta=query_sim_beta,
        max_examples_per_cluster=6,
        text_sim_method=text_sim_method,
        embedding_index=emb_index if query_selection_method not in ("top_k", "random") else None,
    )

    # Attach selection-method identity + per-example acquisition score onto each
    # cluster so the Improver prompt can explain WHY these examples were chosen.
    #
    # Two sources of per-example scores:
    #   - `per_example_scores`: dense vector over all N examples (BatchBALD bald,
    #     BADGE gradient-norm squared, uncertainty importance weight). Covers
    #     cluster members that weren't top-k-selected.
    #   - selected_indices × acquisition_scores: sparse map covering just the
    #     top-k selected examples (all methods). Used for cluster CENTERS whose
    #     center acquisition score is the method-specific batch-level gain.
    _acq_by_idx: Dict[int, float] = {
        int(idx): float(score)
        for idx, score in zip(
            selection.get("selected_indices", []),
            selection.get("acquisition_scores", []),
        )
    }
    _pex_scores = selection.get("per_example_scores") or []
    _pex_by_idx: Dict[int, float] = {}
    if _pex_scores and len(_pex_scores) == len(examples):
        _pex_by_idx = {i: float(_pex_scores[i]) for i in range(len(examples))}

    # Build a percentile-rank map.  Raw scores (e.g. BatchBALD MI) span a narrow
    # numeric band (0.41–0.49), so the magnitude tells the LLM nothing in the
    # prompt.  A rank ("top 3% MI over the dataset") is directly actionable: it
    # says how informative *this example* is relative to the whole pool.
    # rank = fraction of examples with strictly lower score (0 = bottom, 1 = top)
    _pex_rank_by_idx: Dict[int, float] = {}
    if _pex_by_idx:
        _sorted_vals = sorted(_pex_by_idx.values())
        _n_pex = len(_sorted_vals)
        # bisect_left gives #(strictly lower) → percentile in [0, 1)
        import bisect as _bisect
        _pex_rank_by_idx = {
            i: _bisect.bisect_left(_sorted_vals, s) / max(1, _n_pex - 1)
            for i, s in _pex_by_idx.items()
        }
    _acq_rank_by_idx: Dict[int, float] = {}
    if _acq_by_idx:
        _sorted_acq = sorted(_acq_by_idx.values())
        _n_acq = len(_sorted_acq)
        import bisect as _bisect
        _acq_rank_by_idx = {
            i: _bisect.bisect_left(_sorted_acq, s) / max(1, _n_acq - 1)
            for i, s in _acq_by_idx.items()
        }

    for cl in clusters:
        cl["selection_method"] = selection.get("strategy") or selection.get("method") or query_selection_method
        cl["selection_method_label"] = selection.get("method_label")
        cl["selection_score_name"] = selection.get("score_name")
        center_idx = cl.get("center_index")
        if center_idx is not None:
            ci = int(center_idx)
            if ci in _acq_by_idx:
                cl["center_acquisition_score"] = round(_acq_by_idx[ci], 6)
                if ci in _acq_rank_by_idx:
                    cl["center_acquisition_percentile"] = round(_acq_rank_by_idx[ci], 4)
            elif ci in _pex_by_idx:
                cl["center_acquisition_score"] = round(_pex_by_idx[ci], 6)
                if ci in _pex_rank_by_idx:
                    cl["center_acquisition_percentile"] = round(_pex_rank_by_idx[ci], 4)
        # Per-example scores on cluster members. Need the dataset index; clusters
        # only persist ids, so rebuild a map id→index from `examples`.
        # (build once per cluster loop — clusters may reshuffle ids each run.)
        if _pex_by_idx or _acq_by_idx:
            id_to_idx = {str(examples[i].get("id")): i for i in range(len(examples))
                         if examples[i].get("id") is not None}
            for te in cl.get("top_examples", []) or []:
                te_id = te.get("id")
                if te_id is None:
                    continue
                i = id_to_idx.get(str(te_id))
                if i is None:
                    continue
                if i in _acq_by_idx:
                    te["acquisition_score"] = round(_acq_by_idx[i], 6)
                    te["acquisition_score_kind"] = "selected"
                    if i in _acq_rank_by_idx:
                        te["acquisition_percentile"] = round(_acq_rank_by_idx[i], 4)
                elif i in _pex_by_idx:
                    te["acquisition_score"] = round(_pex_by_idx[i], 6)
                    te["acquisition_score_kind"] = "per_example"
                    if i in _pex_rank_by_idx:
                        te["acquisition_percentile"] = round(_pex_rank_by_idx[i], 4)

    # Top-weight examples (useful for baselines / ablations)
    top_weight_indices = sorted(range(n), key=lambda i: float(weights_by_idx[i]), reverse=True)[:max_top_weight_examples]
    top_weight_examples: List[Dict[str, Any]] = []
    for i in top_weight_indices:
        ex = examples[i]
        agg = int(vote_stats["agg_label"][i])
        top_weight_examples.append({
            "index": int(i),
            "id": ex.get("id"),
            "text": str(ex.get("text", ""))[:300],
            "metadata_profile": _feature_profile(ex),
            "vote_depth": int(vote_stats["vote_depth"][i]),
            "disagree": bool(vote_stats["disagree"][i]),
            "agg_label": agg,
            "agg_label_name": LABEL_NAMES.get(agg, "abstain") if agg != ABSTAIN else "abstain",
            "uncertainty_u": round(float(vote_stats["uncertainty_u"][i]), 6),
            "alpha_i": round(float(alpha_i[i]), 6),
            "w_i": round(float(w_i[i]), 6),
        })

    return {
        "paper_method": {
            "objective": query_selection_method,
            "w_i": "w_i = u_i * alpha_i (uncertainty × class_priority)",
            "uncertainty_u": "uncovered=1; disagree=entropy(votes)/log|C|; agree=1/(depth+1)",
            "class_priority": "alpha_c ∝ (1/max(recall_hat_c,epsilon))^gamma estimated on labeled seed",
            "alpha_i": "covered: alpha_{agg_label}; uncovered: E_{pi_hat}[alpha_c]",
            "similarity": f"sim = beta*meta_agreement + (1-beta)*{text_sim_method}",
            "adaptive_budget": "stop when rho=f(S)/W >= rho_target after k_min; cap at k_max",
        },
        "seed_stats": seed_stats,
        "alpha_by_label": {int(k): round(float(v), 6) for k, v in sorted(alpha_by_label.items())},
        "alpha_by_name": alpha_by_name,
        "importance_weight_summary": {
            "W_total": round(float(W_total), 6),
            "expected_alpha_prior": round(float(imp["expected_alpha_prior"]), 6),
            "w_quantiles": _quantiles([float(x) for x in w_i], qs=[0.0, 0.5, 0.9, 0.95, 0.99, 1.0]),
            "u_quantiles": _quantiles([float(x) for x in vote_stats["uncertainty_u"]], qs=[0.0, 0.5, 0.9, 0.95, 0.99, 1.0]),
            "alpha_i_quantiles": _quantiles([float(x) for x in alpha_i], qs=[0.0, 0.5, 0.9, 0.95, 0.99, 1.0]),
            "weight_mass_by_depth_bucket": {k: round(v, 6) for k, v in depth_bucket_mass.items()},
            "weight_mass_disagree": round(float(disagree_mass), 6),
        },
        "selection_config": {
            "query_k_min": int(query_k_min),
            "query_k_max": int(query_k_max),
            "query_rho_target": query_rho_target,
            "query_sim_beta": float(query_sim_beta),
            "text_sim_method": text_sim_method,
            "class_weight_epsilon": float(class_weight_epsilon),
            "class_weight_gamma": float(class_weight_gamma),
            "seed_size": int(seed_size),
            "seed_fraction": float(seed_fraction),
        },
        "selection": selection,
        "selected_examples": selected_examples,
        "clusters": clusters,
        "top_weight_examples": top_weight_examples,
    }


# -----------------------------------------------------------------------
# A. Pool-level summary
# -----------------------------------------------------------------------

def compute_pool_summary(
    annotators: List[Tuple[str, Callable]],
    vote_matrix: List[List[int]],
    examples: List[Dict[str, Any]],
    label_key: str,
) -> Dict[str, Any]:
    n = len(examples)
    n_lfs = len(annotators)

    # per-example: vote depth = how many LFs fire (non-abstain)
    vote_depths: List[int] = []
    is_covered: List[bool] = []
    for i in range(n):
        depth = sum(1 for j in range(n_lfs) if vote_matrix[j][i] != ABSTAIN)
        vote_depths.append(depth)
        is_covered.append(depth > 0)

    n_covered = sum(is_covered)

    # class-level recall (how many of each true class get any vote)
    class_counts: Dict[int, int] = Counter()
    class_covered: Dict[int, int] = Counter()
    for i, ex in enumerate(examples):
        y = ex.get(label_key)
        if y is None:
            continue
        y = int(y)
        class_counts[y] += 1
        if is_covered[i]:
            class_covered[y] += 1

    class_recall = {}
    for c in sorted(class_counts):
        class_recall[LABEL_NAMES.get(c, str(c))] = (
            class_covered[c] / class_counts[c] if class_counts[c] else 0.0
        )

    # LF class balance: how many LFs primarily predict each class
    lf_class_counts: Dict[str, int] = Counter()
    for j in range(n_lfs):
        label_votes = [v for v in vote_matrix[j] if v != ABSTAIN]
        if label_votes:
            majority = Counter(label_votes).most_common(1)[0][0]
            lf_class_counts[LABEL_NAMES.get(majority, str(majority))] += 1

    # vote depth histogram
    depth_hist = dict(sorted(Counter(vote_depths).items()))

    return {
        "n_examples": n,
        "n_lfs": n_lfs,
        "n_covered": n_covered,
        "n_uncovered": n - n_covered,
        "coverage": n_covered / n if n else 0.0,
        "class_counts": {LABEL_NAMES.get(k, str(k)): v for k, v in sorted(class_counts.items())},
        "class_recall": class_recall,
        "lf_class_balance": dict(lf_class_counts),
        "vote_depth_histogram": {str(k): v for k, v in depth_hist.items()},
        "mean_vote_depth_covered": (
            sum(d for d, c in zip(vote_depths, is_covered) if c) / n_covered
            if n_covered else 0.0
        ),
    }


# -----------------------------------------------------------------------
# B. Per-LF diagnostics
# -----------------------------------------------------------------------

def compute_lf_diagnostics(
    annotators: List[Tuple[str, Callable]],
    vote_matrix: List[List[int]],
    examples: List[Dict[str, Any]],
    label_key: str,
) -> Dict[str, Any]:
    n = len(examples)
    n_lfs = len(annotators)

    active_sets: List[Set[int]] = [_active_set(vote_matrix[j]) for j in range(n_lfs)]
    union_all = set().union(*active_sets) if active_sets else set()

    per_lf: List[Dict[str, Any]] = []
    for j, (name, _) in enumerate(annotators):
        fires = len(active_sets[j])
        correct = sum(
            1 for i in active_sets[j]
            if vote_matrix[j][i] == int(examples[i].get(label_key, -99))
        )
        # marginal coverage: examples ONLY this LF covers
        others_union = set().union(*(active_sets[k] for k in range(n_lfs) if k != j)) if n_lfs > 1 else set()
        unique_covered = active_sets[j] - others_union
        marginal_coverage = len(unique_covered) / n if n else 0.0

        predicted_label_counts = Counter(vote_matrix[j][i] for i in active_sets[j])

        per_lf.append({
            "name": name,
            "fires": fires,
            "coverage": fires / n if n else 0.0,
            "precision": correct / fires if fires else None,
            "correct": correct,
            "incorrect": fires - correct,
            "marginal_coverage": marginal_coverage,
            "n_unique_covered": len(unique_covered),
            "predicted_labels": {
                LABEL_NAMES.get(k, str(k)): v
                for k, v in sorted(predicted_label_counts.items())
            },
        })

    # pairwise Jaccard overlap (only report pairs > threshold)
    overlap_threshold = 0.3
    overlapping_pairs: List[Dict[str, Any]] = []
    for i in range(n_lfs):
        for j in range(i + 1, n_lfs):
            inter = len(active_sets[i] & active_sets[j])
            union = len(active_sets[i] | active_sets[j])
            if union == 0:
                continue
            jaccard = inter / union
            if jaccard >= overlap_threshold:
                overlapping_pairs.append({
                    "lf_a": annotators[i][0],
                    "lf_b": annotators[j][0],
                    "jaccard": round(jaccard, 4),
                    "intersection_size": inter,
                })
    overlapping_pairs.sort(key=lambda x: -x["jaccard"])

    # redundancy score: average pairwise Jaccard
    all_jaccards: List[float] = []
    for i in range(n_lfs):
        for j in range(i + 1, n_lfs):
            union = len(active_sets[i] | active_sets[j])
            if union == 0:
                continue
            all_jaccards.append(len(active_sets[i] & active_sets[j]) / union)
    avg_redundancy = sum(all_jaccards) / len(all_jaccards) if all_jaccards else 0.0

    # coverage efficiency: actual union / sum of individual coverages
    sum_individual = sum(len(s) for s in active_sets)
    coverage_efficiency = len(union_all) / sum_individual if sum_individual else 0.0

    return {
        "per_lf": per_lf,
        "overlapping_pairs": overlapping_pairs,
        "avg_pairwise_jaccard": round(avg_redundancy, 4),
        "coverage_efficiency": round(coverage_efficiency, 4),
    }


# -----------------------------------------------------------------------
# C. Coverage gap analysis
# -----------------------------------------------------------------------

def _feature_profile(ex: Dict[str, Any]) -> Dict[str, Any]:
    meta = ex.get("metadata") or {}
    profile: Dict[str, Any] = {}
    for feat in BOOL_FEATURES:
        profile[feat] = bool(meta.get(feat, False))
    profile["text_length"] = _text_length_bucket(ex)
    profile["num_words"] = meta.get("num_words", 0)
    profile["num_exclamation"] = meta.get("num_exclamation", 0)
    return profile


def compute_coverage_gap(
    annotators: List[Tuple[str, Callable]],
    vote_matrix: List[List[int]],
    examples: List[Dict[str, Any]],
    label_key: str,
) -> Dict[str, Any]:
    n = len(examples)
    n_lfs = len(annotators)

    uncovered_indices: List[int] = []
    covered_indices: List[int] = []
    for i in range(n):
        if all(vote_matrix[j][i] == ABSTAIN for j in range(n_lfs)):
            uncovered_indices.append(i)
        else:
            covered_indices.append(i)

    # class breakdown of uncovered
    uncov_class: Dict[str, int] = Counter()
    for i in uncovered_indices:
        y = examples[i].get(label_key)
        if y is not None:
            uncov_class[LABEL_NAMES.get(int(y), str(y))] += 1

    # per-feature coverage gap: for each feature value, what fraction is uncovered
    feature_gaps: List[Dict[str, Any]] = []
    for feat in BOOL_FEATURES + ["text_length"]:
        buckets: Dict[Any, Dict[str, int]] = defaultdict(lambda: {"total": 0, "uncovered": 0})
        for i, ex in enumerate(examples):
            if feat == "text_length":
                val = _text_length_bucket(ex)
            else:
                val = bool((ex.get("metadata") or {}).get(feat, False))
            buckets[val]["total"] += 1
            if i in set(uncovered_indices):
                buckets[val]["uncovered"] += 1
        for val, counts in sorted(buckets.items(), key=lambda x: str(x[0])):
            t = counts["total"]
            u = counts["uncovered"]
            feature_gaps.append({
                "feature": feat,
                "value": val,
                "total": t,
                "uncovered": u,
                "gap_rate": round(u / t, 4) if t else 0.0,
            })

    # compound segments: class x text_length x first_bool_feature
    uncov_set = set(uncovered_indices)
    segment_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "uncovered": 0})
    primary_bool = BOOL_FEATURES[0] if BOOL_FEATURES else None
    for i, ex in enumerate(examples):
        y = ex.get(label_key)
        cls_name = LABEL_NAMES.get(int(y), str(y)) if y is not None else "unknown"
        tl = _text_length_bucket(ex)
        meta = ex.get("metadata") or {}
        if primary_bool:
            feat_val = bool(meta.get(primary_bool, False))
            seg_key = f"{cls_name}|{tl}|{primary_bool}={feat_val}"
        else:
            seg_key = f"{cls_name}|{tl}"
        segment_counts[seg_key]["total"] += 1
        if i in uncov_set:
            segment_counts[seg_key]["uncovered"] += 1

    segments: List[Dict[str, Any]] = []
    for key, counts in segment_counts.items():
        t = counts["total"]
        u = counts["uncovered"]
        parts = key.split("|")
        segments.append({
            "segment_key": key,
            "class": parts[0],
            "text_length": parts[1] if len(parts) > 1 else "",
            "total": t,
            "uncovered": u,
            "gap_rate": round(u / t, 4) if t else 0.0,
        })
    segments.sort(key=lambda s: -s["uncovered"])

    return {
        "n_uncovered": len(uncovered_indices),
        "n_covered": len(covered_indices),
        "uncovered_class_distribution": dict(uncov_class),
        "feature_gaps": feature_gaps,
        "segments": segments,
    }


# -----------------------------------------------------------------------
# D. Fragility analysis
# -----------------------------------------------------------------------

def compute_fragility(
    vote_matrix: List[List[int]],
    examples: List[Dict[str, Any]],
    label_key: str,
) -> Dict[str, Any]:
    n = len(examples)
    n_lfs = len(vote_matrix)

    fragile_indices: List[int] = []
    well_supported_indices: List[int] = []
    for i in range(n):
        depth = sum(1 for j in range(n_lfs) if vote_matrix[j][i] != ABSTAIN)
        if depth == 1:
            fragile_indices.append(i)
        elif depth >= 3:
            well_supported_indices.append(i)

    fragile_class: Dict[str, int] = Counter()
    for i in fragile_indices:
        y = examples[i].get(label_key)
        if y is not None:
            fragile_class[LABEL_NAMES.get(int(y), str(y))] += 1

    return {
        "n_fragile": len(fragile_indices),
        "n_well_supported": len(well_supported_indices),
        "fragile_class_distribution": dict(fragile_class),
        "fragile_fraction_of_covered": (
            len(fragile_indices)
            / (n - sum(1 for i in range(n) if all(vote_matrix[j][i] == ABSTAIN for j in range(n_lfs))))
            if any(any(vote_matrix[j][i] != ABSTAIN for j in range(n_lfs)) for i in range(n))
            else 0.0
        ),
    }


# -----------------------------------------------------------------------
# E. Error analysis
# -----------------------------------------------------------------------

def compute_error_analysis(
    annotators: List[Tuple[str, Callable]],
    vote_matrix: List[List[int]],
    examples: List[Dict[str, Any]],
    label_key: str,
    max_examples: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    n = len(examples)
    n_lfs = len(annotators)
    rng = random.Random(seed)

    # majority vote per example
    def _majority(i: int) -> int:
        active = [vote_matrix[j][i] for j in range(n_lfs) if vote_matrix[j][i] != ABSTAIN]
        if not active:
            return ABSTAIN
        c = Counter(active)
        top = c.most_common()
        if len(top) >= 2 and top[0][1] == top[1][1]:
            return ABSTAIN
        return top[0][0]

    misclassified_indices: List[int] = []   # predicted wrong class
    conflict_indices: List[int] = []

    for i in range(n):
        y = examples[i].get(label_key)
        if y is None:
            continue
        y = int(y)
        pred = _majority(i)
        active_votes = set(vote_matrix[j][i] for j in range(n_lfs) if vote_matrix[j][i] != ABSTAIN)

        if len(active_votes) > 1:
            conflict_indices.append(i)

        if pred != ABSTAIN and pred != y:
            misclassified_indices.append(i)

    def _sample_examples(indices: List[int]) -> List[Dict[str, Any]]:
        sampled = rng.sample(indices, min(max_examples, len(indices)))
        results: List[Dict[str, Any]] = []
        for i in sampled:
            ex = examples[i]
            votes = {annotators[j][0]: vote_matrix[j][i] for j in range(n_lfs) if vote_matrix[j][i] != ABSTAIN}
            results.append({
                "id": ex.get("id"),
                "text": ex.get("text", "")[:300],
                "true_label": ex.get(label_key),
                "true_label_name": LABEL_NAMES.get(int(ex.get(label_key, -1)), "?"),
                "active_votes": votes,
            })
        return results

    # which LFs cause the most misclassifications
    mis_lf_counts: Dict[str, int] = Counter()
    for i in misclassified_indices:
        y = int(examples[i].get(label_key, -1))
        for j in range(n_lfs):
            v = vote_matrix[j][i]
            if v != ABSTAIN and v != y:
                mis_lf_counts[annotators[j][0]] += 1

    return {
        "n_misclassified": len(misclassified_indices),
        "n_conflicts": len(conflict_indices),
        "misclassified_examples": _sample_examples(misclassified_indices),
        "conflict_examples": _sample_examples(conflict_indices),
        "top_error_causing_lfs": dict(Counter(mis_lf_counts).most_common(10)),
        # Backward compat aliases
        "n_false_positives": len(misclassified_indices),
        "n_false_negatives": 0,
        "false_positive_examples": _sample_examples(misclassified_indices),
        "false_negative_examples": [],
    }


# -----------------------------------------------------------------------
# F. Improvement targets: the core "value" scoring
# -----------------------------------------------------------------------

def _extract_distinctive_keywords(
    target_texts: List[str],
    background_texts: List[str],
    top_k: int = 15,
    min_count: int = 3,
) -> List[Tuple[str, float]]:
    """Words overrepresented in target vs background (log-odds-like score)."""
    def _tokenize(text: str) -> Set[str]:
        return {w for w in TOKEN_RE.findall(text.lower()) if len(w) >= 2 and w not in STOPWORDS}

    target_word_counts: Counter = Counter()
    for t in target_texts:
        target_word_counts.update(_tokenize(t))

    bg_word_counts: Counter = Counter()
    for t in background_texts:
        bg_word_counts.update(_tokenize(t))

    n_target = max(len(target_texts), 1)
    n_bg = max(len(background_texts), 1)

    scored: List[Tuple[str, float]] = []
    for word, count in target_word_counts.items():
        if count < min_count:
            continue
        tf_target = count / n_target
        tf_bg = bg_word_counts.get(word, 0) / n_bg
        score = tf_target / (tf_bg + 0.005)
        scored.append((word, round(score, 3)))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def compute_improvement_targets(
    annotators: List[Tuple[str, Callable]],
    vote_matrix: List[List[int]],
    examples: List[Dict[str, Any]],
    label_key: str,
    pool_summary: Dict[str, Any],
    class_weights_override: Optional[Dict[str, float]] = None,
    max_examples_per_target: int = 8,
    representative_strategy: str = "submodular",
    representative_sim_beta: float = 0.5,
    representative_k_min: int = 1,
    representative_rho_target: Optional[float] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    n = len(examples)
    n_lfs = len(vote_matrix)
    rng = random.Random(seed)

    if class_weights_override:
        class_weights: Dict[str, float] = {str(k): float(v) for k, v in class_weights_override.items()}
    else:
        # class weights: inversely proportional to current class recall (oracle, if labels available)
        class_recall = pool_summary.get("class_recall", {})
        class_weights = {}
        for cls_name, recall in class_recall.items():
            class_weights[cls_name] = 1.0 / max(float(recall), 0.01)
        # normalize so they sum to len(class_weights)
        total_w = sum(class_weights.values())
        if total_w > 0:
            factor = len(class_weights) / total_w
            class_weights = {k: round(v * factor, 4) for k, v in class_weights.items()}

    # partition examples
    uncovered_by_class: Dict[str, List[int]] = defaultdict(list)
    covered_by_class: Dict[str, List[int]] = defaultdict(list)
    fragile_by_class: Dict[str, List[int]] = defaultdict(list)
    for i in range(n):
        y = examples[i].get(label_key)
        if y is None:
            continue
        cls_name = LABEL_NAMES.get(int(y), str(y))
        depth = sum(1 for j in range(n_lfs) if vote_matrix[j][i] != ABSTAIN)
        if depth == 0:
            uncovered_by_class[cls_name].append(i)
        else:
            covered_by_class[cls_name].append(i)
            if depth == 1:
                fragile_by_class[cls_name].append(i)

    # build improvement targets per (class x text_length) segment
    segments: Dict[str, List[int]] = defaultdict(list)
    for cls_name, indices in uncovered_by_class.items():
        for i in indices:
            tl = _text_length_bucket(examples[i])
            seg_key = f"uncovered_{cls_name}_{tl}"
            segments[seg_key].append(i)

    # also add fragile segments
    for cls_name, indices in fragile_by_class.items():
        seg_key = f"fragile_{cls_name}"
        segments[seg_key].extend(indices)

    # score and rank targets
    all_covered_texts = [examples[i].get("text", "") for idxs in covered_by_class.values() for i in idxs]

    targets: List[Dict[str, Any]] = []
    for seg_key, indices in segments.items():
        parts = seg_key.split("_", 2)
        target_type = parts[0]          # "uncovered" or "fragile"
        cls_name = parts[1] if len(parts) > 1 else "?"
        detail = parts[2] if len(parts) > 2 else ""

        # value = n_examples × class_weight × type_weight
        type_weight = 1.0 if target_type == "uncovered" else 0.5
        cw = class_weights.get(cls_name, 1.0)
        value = len(indices) * cw * type_weight

        # select representative examples (default: submodular facility location)
        if representative_strategy == "random":
            sampled_idx = rng.sample(indices, min(max_examples_per_target, len(indices)))
            rep_selection = {
                "strategy": "random_sample",
                "k_max": min(max_examples_per_target, len(indices)),
                "selected_indices": sampled_idx,
                "selected_ids": [examples[i].get("id") for i in sampled_idx],
            }
        else:
            rep_selection = facility_location_greedy(
                indices,
                examples,
                k_max=min(max_examples_per_target, len(indices)),
                sim_beta=representative_sim_beta,
                k_min=representative_k_min,
                rho_target=representative_rho_target,
                seed=seed,
            )
            sampled_idx = rep_selection["selected_indices"]
        sampled_examples: List[Dict[str, Any]] = []
        for i in sampled_idx:
            ex = examples[i]
            sampled_examples.append({
                "id": ex.get("id"),
                "text": ex.get("text", "")[:400],
                "true_label": ex.get(label_key),
                "true_label_name": LABEL_NAMES.get(int(ex.get(label_key, -1)), "?"),
                "metadata_profile": _feature_profile(ex),
            })

        # distinctive keywords
        target_texts = [examples[i].get("text", "") for i in indices]
        keywords = _extract_distinctive_keywords(target_texts, all_covered_texts, top_k=15)

        targets.append({
            "segment": seg_key,
            "target_type": target_type,
            "target_class": cls_name,
            "detail": detail,
            "n_examples": len(indices),
            "class_weight": cw,
            "value_score": round(value, 2),
            "distinctive_keywords": keywords,
            "representative_examples": sampled_examples,
            "representative_selection": rep_selection,
        })

    targets.sort(key=lambda t: -t["value_score"])

    # high-level recommendations
    recommendations = _build_recommendations(
        pool_summary, class_weights, uncovered_by_class, fragile_by_class, targets
    )

    return {
        "class_weights": class_weights,
        "targets": targets,
        "recommendations": recommendations,
    }


def _build_recommendations(
    pool_summary: Dict[str, Any],
    class_weights: Dict[str, float],
    uncovered_by_class: Dict[str, List[int]],
    fragile_by_class: Dict[str, List[int]],
    targets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    priority = 0

    # R1: class imbalance
    class_recall = pool_summary.get("class_recall", {})
    lf_balance = pool_summary.get("lf_class_balance", {})
    for cls_name in sorted(class_weights, key=lambda c: -class_weights[c]):
        recall = class_recall.get(cls_name, 0)
        n_lfs_for_class = lf_balance.get(cls_name, 0)
        n_uncov = len(uncovered_by_class.get(cls_name, []))
        if recall < 0.5:
            priority += 1
            recs.append({
                "priority": priority,
                "type": "class_gap",
                "description": (
                    f"Class '{cls_name}' has recall {recall:.1%} with only "
                    f"{n_lfs_for_class} LFs. {n_uncov} examples uncovered. "
                    f"Generate 8-12 new {cls_name}-voting LFs."
                ),
                "target_class": cls_name,
                "current_recall": recall,
                "n_uncovered": n_uncov,
            })

    # R2: overall coverage
    coverage = pool_summary.get("coverage", 0)
    if coverage < 0.7:
        priority += 1
        recs.append({
            "priority": priority,
            "type": "low_coverage",
            "description": (
                f"Overall coverage is {coverage:.1%}. "
                f"Target: >=80%. Generate LFs for the largest uncovered segments."
            ),
            "current_coverage": coverage,
        })

    # R3: fragility
    n_fragile = sum(len(v) for v in fragile_by_class.values())
    n_covered = pool_summary.get("n_covered", 1)
    if n_fragile / max(n_covered, 1) > 0.3:
        priority += 1
        recs.append({
            "priority": priority,
            "type": "fragile_labels",
            "description": (
                f"{n_fragile}/{n_covered} covered examples ({n_fragile/max(n_covered,1):.0%}) "
                f"have only 1 LF firing. Generate corroborating LFs to strengthen these."
            ),
            "n_fragile": n_fragile,
        })

    # R4: redundancy
    avg_jaccard = pool_summary.get("avg_pairwise_jaccard", 0)
    if avg_jaccard > 0.3:
        priority += 1
        recs.append({
            "priority": priority,
            "type": "high_redundancy",
            "description": (
                f"Average pairwise Jaccard overlap is {avg_jaccard:.2f}. "
                f"New LFs should target DIFFERENT examples than existing ones."
            ),
        })

    # R5: top targets
    for t in targets[:3]:
        priority += 1
        kw_str = ", ".join(w for w, _ in t["distinctive_keywords"][:8])
        recs.append({
            "priority": priority,
            "type": "target_segment",
            "description": (
                f"Segment '{t['segment']}': {t['n_examples']} examples "
                f"(value={t['value_score']:.1f}). "
                f"Keywords: [{kw_str}]"
            ),
            "segment": t["segment"],
        })

    return recs


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze annotator pool errors and produce an improvement brief."
    )
    p.add_argument(
        "--task_name", type=str, default="youtube_spam",
        help="Task name (youtube_spam, imdb, agnews). Sets label names and features.",
    )
    p.add_argument(
        "--pool_module", type=Path,
        default=Path("annotators/youtube_spam/initial_pool.py"),
    )
    p.add_argument(
        "--data_path", type=Path,
        default=Path("data/processed/youtube_spam_collection/train.jsonl"),
        help="Dataset to analyze (usually train).",
    )
    p.add_argument(
        "--label_key", type=str, default="true_label",
    )
    p.add_argument(
        "--out_path", type=Path,
        default=Path("runs/youtube_spam/initial/improvement_brief.json"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_examples_per_target", type=int, default=8)
    p.add_argument(
        "--representative_strategy",
        type=str,
        choices=["submodular", "random"],
        default="submodular",
        help="How to select representative examples within each target segment.",
    )
    p.add_argument("--representative_sim_beta", type=float, default=0.5)
    p.add_argument("--representative_k_min", type=int, default=1)
    p.add_argument(
        "--representative_rho_target",
        type=float,
        default=0.0,
        help="If >0, adaptively stop when achieved_rho >= this value.",
    )

    # Paper-aligned submodular query selection (global batch)
    p.add_argument("--query_selection_method", type=str,
        choices=["submodular", "uncertainty", "random", "badge", "batchbald",
                  "a1_margin", "a1_entropy_cb", "a1_v2_kl",
                  "text_cluster_a1err", "disagree_submod", "a1_nbrconflict"],
        default="submodular",
        help="Query selection strategy: submodular (facility location), "
             "uncertainty (top-k by w_i), random, "
             "badge (k-means++ on gradient embeddings, Ash et al. 2020), "
             "batchbald (MI-greedy with MC joint entropy, Kirsch et al. 2019), "
             "or a1-aware: a1_margin/a1_entropy_cb/a1_v2_kl/text_cluster_a1err/"
             "disagree_submod/a1_nbrconflict (see scripts/a1_sel_ablation/a1_signals.py).")
    p.add_argument("--a1_val_path", type=Path, default=None,
                   help="Path to val.jsonl for fitting A1 aggregator (only used by "
                        "a1-aware selection methods). If omitted, train is used as "
                        "self-val (overfits).")
    p.add_argument("--seed_size", type=int, default=200, help="Labeled seed size for estimating class stats (<=0 uses seed_fraction).")
    p.add_argument("--seed_fraction", type=float, default=0.1, help="Used only when seed_size<=0.")
    p.add_argument("--class_weight_epsilon", type=float, default=0.01)
    p.add_argument("--class_weight_gamma", type=float, default=1.0)
    p.add_argument("--query_k_min", type=int, default=12)
    p.add_argument("--query_k_max", type=int, default=48)
    p.add_argument("--query_rho_target", type=float, default=0.85, help="If <=0, disable adaptive stopping and select k_max.")
    p.add_argument("--query_sim_beta", type=float, default=0.5, help="Similarity mix: beta*meta + (1-beta)*text_sim.")
    p.add_argument("--text_sim_method", type=str, choices=["jaccard", "tfidf", "semantic", "hybrid"], default="jaccard",
                   help="Text similarity method: 'jaccard' (token overlap), 'tfidf' (TF-IDF cosine), 'semantic' (precomputed embeddings), or 'hybrid' (TF-IDF + LF coverage, adaptive alpha).")
    p.add_argument("--embeddings_path", type=str, default=None,
                   help="Path to .npz file with precomputed embeddings (required when text_sim_method=semantic).")
    p.add_argument("--generation_memory", type=Path, default=None,
                   help="Path to cross-iteration memory JSON.")
    p.add_argument("--carryover_lambda", type=float, default=0.1,
                   help="Lambda for carryover-aware BALD boost: "
                        "bald_i *= (1 + lambda * min(age_i, cap)). 0 disables.")
    p.add_argument("--carryover_age_cap", type=int, default=4,
                   help="Cap on carryover age before the boost saturates; "
                        "prevents indefinitely boosting out-of-reach points.")
    p.add_argument("--class_balance_floor", type=int, default=0,
                   help="Per-class minimum slots in the selected batch. "
                        "0 disables (default). Applies to BatchBALD and "
                        "facility-location greedy. Pseudo-label per example "
                        "is the pool's majority-vote aggregate; ABSTAIN ignored.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # Override module-level globals from task config
    global LABEL_NAMES, BOOL_FEATURES
    from src.tasks.configs import get_task_config
    try:
        task_cfg = get_task_config(args.task_name)
        LABEL_NAMES = dict(task_cfg.label_names)
        BOOL_FEATURES = list(task_cfg.bool_features)
        print(f"Task: {args.task_name} — labels={LABEL_NAMES}, features={BOOL_FEATURES}")
    except KeyError:
        print(f"WARNING: unknown task_name={args.task_name!r}, using defaults")

    print(f"Loading pool from {args.pool_module} ...")
    mod = load_pool_module(args.pool_module)
    annotators = get_annotators(mod)
    print(f"  {len(annotators)} annotators loaded")

    print(f"Reading data from {args.data_path} ...")
    examples = read_jsonl(args.data_path)
    print(f"  {len(examples)} examples")

    print("Running LFs ...")
    vote_matrix = run_pool(annotators, examples)

    print("Computing pool summary ...")
    pool_summary = compute_pool_summary(annotators, vote_matrix, examples, args.label_key)
    print(f"  coverage={pool_summary['coverage']:.3f}  "
          f"class_recall={pool_summary['class_recall']}")

    print("Computing LF diagnostics ...")
    lf_diag = compute_lf_diagnostics(annotators, vote_matrix, examples, args.label_key)
    print(f"  avg_jaccard={lf_diag['avg_pairwise_jaccard']}  "
          f"coverage_efficiency={lf_diag['coverage_efficiency']}")
    # Make redundancy visible to downstream recommendation logic
    pool_summary["avg_pairwise_jaccard"] = lf_diag.get("avg_pairwise_jaccard", 0)

    print("Computing coverage gap ...")
    cov_gap = compute_coverage_gap(annotators, vote_matrix, examples, args.label_key)
    print(f"  uncovered: {cov_gap['uncovered_class_distribution']}")

    print("Computing fragility ...")
    fragility = compute_fragility(vote_matrix, examples, args.label_key)
    print(f"  fragile={fragility['n_fragile']}  well_supported={fragility['n_well_supported']}")

    print("Computing error analysis ...")
    errors = compute_error_analysis(annotators, vote_matrix, examples, args.label_key, seed=args.seed)
    print(f"  FP={errors['n_false_positives']}  FN={errors['n_false_negatives']}  "
          f"conflicts={errors['n_conflicts']}")

    # Build carryover_age_by_id from shared_memory's selection_history, if any.
    # carryover_age_i = number of recent iterations in which id i appeared in
    # `high_info_carryover_ids` (i.e., was selected by the selector and remained
    # uncovered by the next iter's pool). Used to boost BatchBALD MI by
    # (1 + lambda * age).
    _carryover_age_by_id: Dict[str, int] = {}
    if args.generation_memory and args.generation_memory.exists():
        try:
            _mem = json.loads(args.generation_memory.read_text())
            _sh = _mem.get("selection_history", {}) or {}
            # Consider the last N iters of history (skip iter 0 which has no carryover)
            _keys = sorted([int(k) for k in _sh.keys() if str(k).isdigit()])
            for _k in _keys:
                _ids = (_sh.get(str(_k)) or _sh.get(_k) or {}).get("high_info_carryover_ids", []) or []
                for _i in _ids:
                    _carryover_age_by_id[str(_i)] = _carryover_age_by_id.get(str(_i), 0) + 1
            if _carryover_age_by_id:
                _max_age = max(_carryover_age_by_id.values())
                print(f"  Carryover-age boost: {len(_carryover_age_by_id)} ids "
                      f"(max_age={_max_age}, lambda={args.carryover_lambda}, "
                      f"cap={args.carryover_age_cap})")
        except Exception as _e:
            print(f"  Warning: could not build carryover map: {_e}")

    # Load val rows if a1-aware selection requested
    _a1_val_rows = None
    if args.a1_val_path is not None and args.a1_val_path.exists():
        _a1_val_rows = read_jsonl(args.a1_val_path)
        print(f"  Loaded {len(_a1_val_rows)} val rows from {args.a1_val_path} for A1 fit")

    print(f"Computing query selection (method={args.query_selection_method}) ...")
    query_selection = compute_query_selection_paper_method(
        annotators,
        vote_matrix,
        examples,
        label_key=args.label_key,
        seed=args.seed,
        seed_size=args.seed_size,
        seed_fraction=args.seed_fraction,
        class_weight_epsilon=args.class_weight_epsilon,
        class_weight_gamma=args.class_weight_gamma,
        query_k_min=args.query_k_min,
        query_k_max=args.query_k_max,
        query_rho_target=(args.query_rho_target if args.query_rho_target > 0 else None),
        query_sim_beta=args.query_sim_beta,
        text_sim_method=args.text_sim_method,
        embeddings_path=args.embeddings_path,
        query_selection_method=args.query_selection_method,
        carryover_age_by_id=_carryover_age_by_id or None,
        carryover_lambda=float(args.carryover_lambda),
        carryover_age_cap=int(args.carryover_age_cap),
        class_balance_floor=int(args.class_balance_floor),
        a1_val_rows=_a1_val_rows,
    )
    sel = query_selection.get("selection", {}) or {}
    print(
        f"  selected_k={len(sel.get('selected_indices', []))}  "
        f"achieved_rho={sel.get('achieved_rho')}  "
        f"alpha={query_selection.get('alpha_by_name')}"
    )

    print("Computing improvement targets ...")
    targets = compute_improvement_targets(
        annotators,
        vote_matrix,
        examples,
        args.label_key,
        pool_summary,
        class_weights_override=(query_selection.get("alpha_by_name") or None),
        max_examples_per_target=args.max_examples_per_target,
        representative_strategy=args.representative_strategy,
        representative_sim_beta=args.representative_sim_beta,
        representative_k_min=args.representative_k_min,
        representative_rho_target=(args.representative_rho_target if args.representative_rho_target > 0 else None),
        seed=args.seed,
    )
    print(f"  {len(targets['targets'])} targets, top recommendations:")
    for rec in targets["recommendations"]:
        print(f"    P{rec['priority']}: {rec['description']}")

    # Load cross-iteration memory summary for the brief
    memory_summary: Optional[Dict[str, Any]] = None
    if args.generation_memory and args.generation_memory.exists():
        try:
            gen_memory = json.loads(args.generation_memory.read_text())
            iter_history = gen_memory.get("iteration_history", [])
            class_diff = gen_memory.get("class_difficulty", {})
            memory_summary = {
                "iterations_tracked": len(iter_history),
                "class_difficulty": {
                    cls: {
                        "difficulty": info.get("difficulty"),
                        "best_f1": info.get("best_f1"),
                        "stagnant_iters": info.get("stagnant_iters", 0),
                    }
                    for cls, info in class_diff.items()
                },
                "coverage_trajectory": [
                    {"iter": h.get("iter"), "coverage": h.get("coverage")}
                    for h in iter_history[-5:]
                ],
                "stagnant_classes": [
                    cls for cls, info in class_diff.items()
                    if info.get("stagnant_iters", 0) >= 2
                ],
            }
            print(f"  Memory: {len(iter_history)} iterations tracked, "
                  f"{len(memory_summary['stagnant_classes'])} stagnant classes")
        except Exception as e:
            print(f"  Warning: could not load memory: {e}")

    # Assemble brief
    brief = {
        "created_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).replace(microsecond=0).isoformat(),
        "pool_module": str(args.pool_module),
        "data_path": str(args.data_path),
        "pool_summary": pool_summary,
        "lf_diagnostics": lf_diag,
        "coverage_gap": cov_gap,
        "fragility": fragility,
        "error_analysis": errors,
        "query_selection": query_selection,
        "improvement_targets": targets,
    }
    if memory_summary is not None:
        brief["memory_summary"] = memory_summary
    write_json(args.out_path, brief)
    print(f"\nWrote improvement brief: {args.out_path}")


if __name__ == "__main__":
    main()
