"""Diversity metrics and the sentence embedder.

Definitions, in the order the report uses them.

Distinct-n (Li et al., NAACL 2016). For a response set S,

    distinct_n(S) = |unique n-grams in S| / |n-gram tokens in S|

computed over the *pooled* n-grams of all responses in the set, so it measures
variety across the ten samples rather than within any one of them. It is
mechanically length-biased: the denominator grows with total length while the
numerator saturates, so a verbose arm scores lower even at equal variety. We
therefore always log `total_tokens` next to it and never compare distinct-n
between arms whose lengths differ materially without saying so.

Semantic diversity. With L2-normalised embeddings e_i of the k responses,

    semantic_diversity(S) = mean over i<j of (1 - cos(e_i, e_j))

The i<j restriction drops the diagonal. Including it would mix k zero
self-distances into the mean and shrink every score by a factor (k-1)/k, and
would do so *unequally* if k ever differed across arms.

Partition proxy. NoveltyBench's headline number counts how many of ten
responses are *meaningfully* distinct, using a trained partition model. We do
not have that model here, so we approximate it: walk the responses in order and
open a new equivalence class whenever a response is below the similarity
threshold against every class representative seen so far. This is a crude
single-pass, order-dependent stand-in. It is named `*_proxy` everywhere, and a
number from it must never be reported as a NoveltyBench score.

Reference-relative novelty. Diversity asks how much a set varies internally;
novelty asks whether a response adds something *relative to a specified
reference set* (see ESTR4998/background/DIVERSITY_NOVELTY_QUALITY.md). Here the reference
R_p for prompt p is an independent sample of k responses from the generator
under `default` decoding and the verbatim prompt, drawn with its own seed. With
L2-normalised embeddings,

    novelty(s) = 1 - max over r in R_p of cos(e_s, e_r)

i.e. the cosine distance from a response to its *nearest* reference response,
and the set score is the mean over the set. `novel_share_proxy` is the share of
responses whose nearest-reference similarity is below the partition threshold:
the embedding analogue of NoveltyBench's "new equivalence class", measured
against what the model says by default instead of against earlier samples.

Because R_p is independent of every arm, the `default` arm's own novelty is
generally not zero: it is the novelty one fresh default sample has against
another. That is a *baseline*, not a floor -- another arm can score below it,
and a default set that happens to repeat the reference answers can score zero.
Novelty here is relative to this model's default behaviour only -- not to a
human corpus, to the training data, or to history.

Kernel-entropy diversity (since metric v0.4). Both scores are the exponential
of an entropy of the normalised kernel matrix K/k of the k responses. They read
as an "effective number of distinct responses": 1 for k identical responses and
k for k mutually unrelated ones.

  * Vendi score (Friedman & Dieng, 2023): exp(-Σ λ_i log λ_i) over the
    eigenvalues λ of K/k, with the cosine kernel K = E Eᵀ. It has no
    bandwidth. Its order-1 entropy is the matrix-based von Neumann entropy of
    Wu & Farnia (literature/meetings).
  * RKE score (Jalali, Li & Farnia, NeurIPS 2023; used for guidance in SPARKE):
    1 / ||K/k||_F², the exponential of order-2 Renyi entropy, with a Gaussian kernel
    exp(-||e_i - e_j||² / 2σ²). The bandwidth σ sets the scale at which two
    responses count as different. It is a fixed config value
    (`rke_bandwidth`), never tuned on the data being scored.

Like every embedding metric here they inherit MiniLM's topic-level view of
similarity. Unlike the functional classifier, they read the whole response.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constraints import CONSTRAINTS_VERSION
from .functional import FUNCTIONAL_VERSION

METRIC_VERSIONS = {
    "exact_duplicate_rate": "v0.3/whitespace-collapsed-string-equality",
    "distinct_n": "v0.1/pooled-set-level/regex-word-punct-lowercase",
    "semantic_diversity": "v0.1/mean-upper-triangle-cosine-distance",
    "partition_count_proxy": "v0.1/greedy-first-fit-threshold",
    "reference_novelty": "v0.1/mean-nearest-reference-cosine-distance",
    "novel_share_proxy": "v0.1/share-below-threshold-vs-reference",
    # v0.2: long texts are embedded in 256-token windows and pooled over every
    # token; v0.1 silently truncated at 256 tokens (see DESIGN.md section 12).
    "embedding": "v0.2/minilm-chunked-mean-pooling-l2-normalised",
    "functional": FUNCTIONAL_VERSION,
    # v0.3 adds `ai_disclaimer`, which v0.2's refusal markers missed (DESIGN.md section 17).
    # v0.4 (9944df2): mixed_script over the main BMP script blocks; v0.5 widens
    # it to Armenian, all Indic, Georgian, Ethiopic and supplementary-plane CJK.
    "degeneracy_flags": "v0.5/empty-short-repeat-leak-refusal-disclaimer-mixedscript-wide",
    "vendi_score": "v0.1/cosine-kernel-exp-shannon-entropy",
    "rke_score": "v0.1/gaussian-kernel-inverse-squared-frobenius",
    "constraints": CONSTRAINTS_VERSION,
}

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-cased word and punctuation tokens.

    Deliberately not the model tokenizer: lexical diversity should not change
    when the generator is swapped for one with a different vocabulary.
    """
    return _TOKEN_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Unique / total pooled n-grams. Returns 0.0 for an empty set."""
    pooled: list[tuple[str, ...]] = []
    for text in texts:
        pooled.extend(ngrams(tokenize(text), n))
    if not pooled:
        return 0.0
    return len(set(pooled)) / len(pooled)


def total_tokens(texts: Sequence[str]) -> int:
    return sum(len(tokenize(text)) for text in texts)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity; zero vectors yield similarity 0."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D array of embeddings, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    unit = matrix / safe
    unit[np.squeeze(norms == 0.0, axis=1)] = 0.0
    return np.clip(unit @ unit.T, -1.0, 1.0)


def mean_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
    """Mean cosine distance over distinct pairs (diagonal excluded)."""
    similarity = cosine_similarity_matrix(embeddings)
    count = similarity.shape[0]
    if count < 2:
        return 0.0
    upper = np.triu_indices(count, k=1)
    return float(np.mean(1.0 - similarity[upper]))


def partition_proxy_labels(embeddings: np.ndarray, threshold: float) -> list[int]:
    """Class label per response under the greedy first-fit embedding proxy."""
    similarity = cosine_similarity_matrix(embeddings)
    representatives: list[int] = []
    labels: list[int] = []
    for index in range(similarity.shape[0]):
        for label, rep in enumerate(representatives):
            if similarity[index, rep] >= threshold:
                labels.append(label)
                break
        else:
            labels.append(len(representatives))
            representatives.append(index)
    return labels


def partition_count_proxy(embeddings: np.ndarray, threshold: float) -> int:
    """Greedy first-fit count of near-distinct responses. See module docstring."""
    return len(set(partition_proxy_labels(embeddings, threshold)))


def nearest_reference_similarity(
    embeddings: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """For each response, its highest cosine similarity to any reference response."""
    if len(reference) == 0:
        raise ValueError("reference set is empty; novelty is undefined without one")
    if len(embeddings) == 0:
        return np.zeros(0)
    stacked = np.concatenate([np.asarray(embeddings), np.asarray(reference)], axis=0)
    similarity = cosine_similarity_matrix(stacked)
    cross = similarity[: len(embeddings), len(embeddings) :]
    return cross.max(axis=1)


def reference_novelty(embeddings: np.ndarray, reference: np.ndarray) -> float:
    """Mean distance from each response to its nearest reference response."""
    nearest = nearest_reference_similarity(embeddings, reference)
    return float(np.mean(1.0 - nearest)) if len(nearest) else 0.0


def novel_share_proxy(embeddings: np.ndarray, reference: np.ndarray, threshold: float) -> float:
    """Share of responses below `threshold` similarity to every reference response."""
    nearest = nearest_reference_similarity(embeddings, reference)
    return float(np.mean(nearest < threshold)) if len(nearest) else 0.0


def vendi_score(embeddings: np.ndarray) -> float:
    """exp(Shannon entropy) of the eigenvalues of K/k, cosine kernel. See module docstring."""
    count = len(embeddings)
    if count == 0:
        return 0.0
    eigenvalues = np.clip(np.linalg.eigvalsh(cosine_similarity_matrix(embeddings) / count), 0.0, None)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))


def rke_score(embeddings: np.ndarray, bandwidth: float) -> float:
    """1 / ||K/k||_F² with a Gaussian kernel of width `bandwidth`. See module docstring."""
    if bandwidth <= 0:
        raise ValueError("RKE bandwidth must be > 0")
    count = len(embeddings)
    if count == 0:
        return 0.0
    vectors = np.asarray(embeddings, dtype=np.float64)
    squared = np.sum((vectors[:, None, :] - vectors[None, :, :]) ** 2, axis=-1)
    kernel = np.exp(-squared / (2.0 * bandwidth ** 2))
    return float(1.0 / np.sum((kernel / count) ** 2))


def exact_duplicate_rate(texts: Sequence[str]) -> float:
    """Share of responses that repeat an earlier response, ignoring whitespace layout.

    Runs of whitespace collapse to one space, so "atoms?\n\nBecause" and
    "atoms? Because" are the same answer; case and punctuation still count,
    unlike v0.1, which lowercased and re-tokenised. v0.2 compared raw strings
    and missed such layout-only repeats.
    """
    if not texts:
        return 0.0
    seen: set[str] = set()
    duplicates = 0
    for text in texts:
        key = " ".join(text.split())
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates / len(texts)


# --- reward-hacking / degeneracy flags -----------------------------------
LEAK_MARKERS = (
    "creative and diverse", "diverse answer", "creative and unusual", "unusual response",
    "as instructed", "you asked me to",
)
# Openings of a declined task. A refusal is kept and labelled, never dropped
# (INITIAL_RESEARCH_SCOPE.md, "Response contract"); like the other flags it
# only decides what a human reads first.
REFUSAL_MARKERS = (
    "i'm sorry, but i can", "i am sorry, but i can", "i can't help with",
    "i cannot help with", "i can't assist", "i cannot assist", "i'm unable to",
    "i am unable to", "i won't be able to", "i do not engage in",
)
# Meta-commentary about being an AI ("As an AI language model, I don't have
# personal experiences..."). The response usually goes on to answer, so it is
# not a refusal, but it breaks the response contract's "one response to the
# original task". 15 v0.4.0 responses open this way and the refusal flag missed
# every one (archive/v0_4/RESULTS.md).
DISCLAIMER_MARKERS = (
    "as an ai language model", "as an ai assistant", "as an ai,", "as a language model",
    "as an artificial intelligence", "i don't have personal experiences",
    "i do not have personal experiences",
)


# Letters of non-Latin scripts in an English task: untruncated sampling drifts
# into multi-script word salad (97 of its 300 v0.5.0 responses), which the
# other flags mostly miss and the judge scores 4 or more.
# Cyrillic, Armenian, Hebrew, Arabic, the Indic scripts (Devanagari to
# Sinhala), Thai, Georgian, Hangul, Ethiopic, kana, and CJK including the
# supplementary-plane extensions.
MIXED_SCRIPT = re.compile(
    "[\u0400-\u04ff\u0530-\u058f\u0590-\u05ff\u0600-\u06ff\u0900-\u0dff\u0e00-\u0e7f"
    "\u10a0-\u10ff\u1100-\u11ff\u1200-\u137f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    "\uac00-\ud7af\U00020000-\U0002fa1f]")


def degeneracy_flags(text: str, min_tokens: int = 5) -> dict[str, bool]:
    """Cheap signals that a high diversity score may be an artefact.

    These do not prove anything. They decide which outputs a human reads first.
    Curly apostrophes are straightened before markers are matched.
    """
    tokens = tokenize(text)
    counts = Counter(ngrams(tokens, 3))
    most_common = counts.most_common(1)[0][1] if counts else 0
    lowered = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    return {
        "empty": not tokens,
        "very_short": 0 < len(tokens) < min_tokens,
        "repetitive": most_common >= 4,
        "instruction_leak": any(marker in lowered for marker in LEAK_MARKERS),
        "refusal": any(marker in lowered[:200] for marker in REFUSAL_MARKERS),
        "ai_disclaimer": any(marker in lowered for marker in DISCLAIMER_MARKERS),
        "mixed_script": bool(MIXED_SCRIPT.search(text)),
    }


@dataclass
class SetMetrics:
    distinct_1: float
    distinct_2: float
    semantic_diversity: float
    partition_count_proxy: int
    exact_duplicate_rate: float
    total_tokens: int
    mean_tokens_per_response: float
    reference_novelty: float | None = None
    novel_share_proxy: float | None = None
    # NoveltyBench v1.0 functional metrics (functional.py); None when not run.
    functional_classes: int | None = None
    functional_novel_share: float | None = None
    functional_window_truncated_share: float | None = None
    # Kernel-entropy diversity (metric v0.4); RKE is None without a bandwidth.
    vendi_score: float | None = None
    rke_score: float | None = None
    # Share of responses meeting every countable constraint of the prompt
    # (constraints.py); None when the prompt states none.
    constraint_satisfaction: float | None = None

    def as_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def set_metrics(
    texts: Sequence[str],
    embeddings: np.ndarray,
    partition_threshold: float,
    reference: np.ndarray | None = None,
    rke_bandwidth: float | None = None,
) -> SetMetrics:
    """All set-level metrics for one (prompt, condition, replicate) response set.

    Novelty fields stay None when no reference set is given, never 0, so a run
    without references cannot be mistaken for one with zero novelty. Likewise
    `rke_score` stays None unless a bandwidth is configured.
    """
    if len(texts) != len(embeddings):
        raise ValueError(
            f"{len(texts)} responses but {len(embeddings)} embeddings; "
            "a mismatch here silently mis-pairs text with vectors"
        )
    tokens = total_tokens(texts)
    return SetMetrics(
        distinct_1=distinct_n(texts, 1),
        distinct_2=distinct_n(texts, 2),
        semantic_diversity=mean_pairwise_cosine_distance(embeddings),
        partition_count_proxy=partition_count_proxy(embeddings, partition_threshold),
        exact_duplicate_rate=exact_duplicate_rate(texts),
        total_tokens=tokens,
        mean_tokens_per_response=tokens / len(texts) if texts else 0.0,
        reference_novelty=(
            None if reference is None else reference_novelty(embeddings, reference)
        ),
        novel_share_proxy=(
            None
            if reference is None
            else novel_share_proxy(embeddings, reference, partition_threshold)
        ),
        vendi_score=vendi_score(embeddings),
        rke_score=None if rke_bandwidth is None else rke_score(embeddings, rke_bandwidth),
    )


class Embedder:
    """all-MiniLM-L6-v2 with mean pooling and L2 normalisation, over the whole text.

    For a text of at most `window - 2` = 254 word pieces this reproduces the
    `sentence-transformers` pipeline for this checkpoint (its `modules.json`
    chains the transformer, mean pooling per `1_Pooling/config.json`, and a
    Normalize module), using plain `transformers`. `test_embedder_reference.py`
    checks it against vectors the library itself produced. Longer texts are
    where the two part ways, and the reason this class exists: the library
    truncates at 256 positions, and pooling across windows needs the word-piece
    ids and per-token vectors that it does not expose.

    Mean pooling is computed over the attention mask, so a padded batch gives
    the same vectors as an unpadded one up to float error.
    `test_embedder_batching.py` checks that.

    Long texts. The checkpoint accepts at most `window` = 256 positions
    including [CLS] and [SEP]. Truncating there (what v0.1 did) meant the
    embedder never saw the second half of long answers, and the arm that writes
    the longest answers (`naive_diverse`) lost the most. Instead, a text is cut
    into consecutive windows of `window - 2` word pieces, each window is encoded
    separately, and the token vectors of all windows are mean-pooled together
    before normalising. That is exactly the v0.1 vector for any text that fits
    in one window. For a longer text every word piece carries equal weight, but
    each window also adds its own [CLS] and [SEP] to the mean, and a window
    boundary can fall inside a word. Windows do not see each other, so
    cross-window context is lost; for embeddings used only to compare topic and
    content this is the usual, accepted approximation.
    """

    def __init__(
        self,
        model_id: str,
        revision: str,
        device: str = "auto",
        batch_size: int = 64,
        window: int = 256,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        resolved = device
        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = resolved
        self.batch_size = batch_size
        self.window = window
        self.model_id = model_id
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        # Texts are tokenised whole and then split into windows, so the
        # tokenizer's "sequence too long" warning does not apply; lift the limit
        # it checks against. The model itself never sees more than `window`.
        self.tokenizer.model_max_length = 1 << 30
        # float32: the embedder is cheap, and bf16 rounding would add noise to a
        # metric we compare across arms at the third decimal place.
        self.model = AutoModel.from_pretrained(
            model_id, revision=revision, dtype=torch.float32
        ).to(resolved)
        self.model.eval()

    def _windows(self, text: str) -> list[list[int]]:
        """Word-piece ids of `text`, split into windows that each fit the model."""
        ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        width = self.window - 2  # room for [CLS] and [SEP]
        chunks = [ids[start : start + width] for start in range(0, len(ids), width)] or [[]]
        cls, sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        return [[cls, *chunk, sep] for chunk in chunks]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        windows: list[list[int]] = []
        owner: list[int] = []
        for index, text in enumerate(texts):
            for window in self._windows(text):
                windows.append(window)
                owner.append(index)
        hidden_size = self.model.config.hidden_size
        if not windows:
            return np.zeros((0, hidden_size), dtype=np.float32)

        # Sum of masked token vectors and token count per text, accumulated
        # across windows and batches; dividing at the end gives one mean over
        # every token of the text, however it was split.
        sums = torch.zeros((len(texts), hidden_size), dtype=torch.float64)
        counts = torch.zeros((len(texts), 1), dtype=torch.float64)
        pad_id = self.tokenizer.pad_token_id
        for start in range(0, len(windows), self.batch_size):
            batch = windows[start : start + self.batch_size]
            width = max(len(window) for window in batch)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attention = torch.zeros((len(batch), width), dtype=torch.long)
            for row, window in enumerate(batch):
                input_ids[row, : len(window)] = torch.tensor(window, dtype=torch.long)
                attention[row, : len(window)] = 1
            input_ids, attention = input_ids.to(self.device), attention.to(self.device)
            with torch.no_grad():
                hidden = self.model(input_ids=input_ids, attention_mask=attention).last_hidden_state
            mask = attention.unsqueeze(-1).to(hidden.dtype)
            window_sums = (hidden * mask).sum(dim=1).double().cpu()
            window_counts = mask.sum(dim=1).double().cpu()
            for row, text_index in enumerate(owner[start : start + len(batch)]):
                sums[text_index] += window_sums[row]
                counts[text_index] += window_counts[row]
        pooled = sums / counts.clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.float().numpy()
