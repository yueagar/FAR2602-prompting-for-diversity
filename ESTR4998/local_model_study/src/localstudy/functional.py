"""Functional equivalence: NoveltyBench v1.0's classifier and partition.

`ESTR4998/background/DIVERSITY_NOVELTY_QUALITY.md` defines diversity as *meaningful*
variation, and NoveltyBench operationalises it as functional equivalence: two
responses are equivalent when a user who has seen one would gain little from
the other. The embedding metrics in `metrics.py` only approximate that, so
protocol v0.4.0 adds NoveltyBench's own v1.0 instrument, reproduced from
`novelty-bench/novelty-bench` `src/partition.py`:

* classifier `yimingzhang/deberta-v3-large-generation-similarity` with the
  `microsoft/deberta-v3-large` tokenizer;
* input `[CLS] a[:128] [SEP] b[:128] [SEP]`, token types 0 through the first
  [SEP] and 1 after it; the prompt is not shown to the classifier;
* `a` and `b` are equivalent when softmax(logits)[1] > 0.102, and identical
  strings are equivalent without asking the classifier;
* partition: walk the responses in order; each joins the first class whose
  *head* (first member) it is equivalent to, otherwise it opens a new class.
  `distinct_k` is the number of classes.

Functional novelty applies the same test against the prompt's reference pool:
a response is functionally novel when no reference response is equivalent to
it (reference as `a`, response as `b`, the head orientation of the partition).

Known limits, all reported rather than hidden:

* **128-token window.** Each response is cut to its first 128 DeBERTa tokens,
  as in training, so two stories that share an opening and diverge later are
  scored as equivalent. `functional_window_truncated_share` reports how many
  responses exceeded the window. NoveltyBench's own README warns about this
  for long-form generation; five-sentence stories are near the limit.
* **Threshold margin.** 0.102 is NoveltyBench's tuned cut-off; unrelated
  creative texts can score close to it, so small classifier shifts flip
  verdicts. The human pair-annotation sheet (`validity.py`) measures how well
  the classifier agrees with people on this domain.
* **Licence.** The classifier's model card states no licence; it is used here
  for research measurement only, and its revision is pinned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

WINDOW = 128
THRESHOLD = 0.102
FUNCTIONAL_VERSION = "v0.1/novelty-bench-v1.0-classifier-head-partition"

EquivalenceFn = Callable[[Sequence[tuple[str, str]]], list[bool]]


def partition_labels(texts: Sequence[str], equivalent: EquivalenceFn) -> list[int]:
    """NoveltyBench's head partition; `equivalent` answers (head, response) pairs.

    Pairs are only needed for head < response, so all candidate pairs are
    asked in one batch and the walk then reads the answers. That gives the same
    partition as asking lazily, because a pair's answer does not depend on the
    order it is asked in.
    """
    count = len(texts)
    pairs = [(a, b) for b in range(count) for a in range(b)]
    answers = dict(zip(pairs, equivalent([(texts[a], texts[b]) for a, b in pairs]))) if pairs else {}
    heads: list[int] = []
    labels = [-1] * count
    for index in range(count):
        for label, head in enumerate(heads):
            if answers[(head, index)]:
                labels[index] = label
                break
        else:
            labels[index] = len(heads)
            heads.append(index)
    return labels


def novel_flags(
    texts: Sequence[str], reference: Sequence[str], equivalent: EquivalenceFn
) -> list[bool]:
    """True where no reference response is functionally equivalent to the response."""
    if not reference:
        raise ValueError("reference set is empty; functional novelty is undefined without one")
    pairs = [(ref, text) for text in texts for ref in reference]
    answers = equivalent(pairs)
    width = len(reference)
    return [not any(answers[i * width : (i + 1) * width]) for i in range(len(texts))]


@dataclass
class FunctionalMetrics:
    functional_classes: int
    functional_labels: list[int]
    functional_novel_share: float | None
    functional_novel: list[bool] | None
    functional_window_truncated_share: float

    def set_level(self) -> dict[str, float | int | None]:
        return {
            "functional_classes": self.functional_classes,
            "functional_novel_share": self.functional_novel_share,
            "functional_window_truncated_share": self.functional_window_truncated_share,
        }


class FunctionalJudge:
    """The pinned NoveltyBench v1.0 classifier, batched."""

    def __init__(self, classifier, tokenizer, device: str = "auto", batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.classifier_spec, self.tokenizer_spec = classifier, tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer.model_id, revision=tokenizer.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            classifier.model_id, revision=classifier.revision,
            dtype=getattr(torch, classifier.dtype),
        ).to(device)
        self.model.eval()

    @property
    def resolved_revision(self) -> str:
        return getattr(self.model.config, "_commit_hash", None) or self.classifier_spec.revision

    def _pieces(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, truncation=True, max_length=WINDOW, add_special_tokens=False)

    def exceeds_window(self, text: str) -> bool:
        return len(self.tokenizer.encode(text, add_special_tokens=False)) > WINDOW

    def probabilities(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """P(equivalent) for each (a, b), exactly as NoveltyBench builds the input."""
        torch = self._torch
        tok = self.tokenizer
        rows = []
        for a, b in pairs:
            first = [tok.cls_token_id, *self._pieces(a), tok.sep_token_id]
            ids = first + [*self._pieces(b), tok.sep_token_id]
            rows.append((ids, [0] * len(first) + [1] * (len(ids) - len(first))))
        out: list[float] = []
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            width = max(len(ids) for ids, _ in batch)
            input_ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
            types = torch.zeros((len(batch), width), dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, (ids, tt) in enumerate(batch):
                input_ids[row, : len(ids)] = torch.tensor(ids)
                types[row, : len(tt)] = torch.tensor(tt)
                mask[row, : len(ids)] = 1
            with torch.inference_mode():
                logits = self.model(
                    input_ids=input_ids.to(self.device),
                    token_type_ids=types.to(self.device),
                    attention_mask=mask.to(self.device),
                ).logits.float()
            out.extend(logits.softmax(-1)[:, 1].cpu().tolist())
        return out

    def equivalent(self, pairs: Sequence[tuple[str, str]]) -> list[bool]:
        verdicts: list[bool | None] = [True if a == b else None for a, b in pairs]
        ask = [i for i, v in enumerate(verdicts) if v is None]
        for i, p in zip(ask, self.probabilities([pairs[i] for i in ask])):
            verdicts[i] = p > THRESHOLD
        return [bool(v) for v in verdicts]

    def measure(self, texts: Sequence[str], reference: Sequence[str] | None) -> FunctionalMetrics:
        # A novelty share has no denominator without observed responses. Runs
        # require k >= 1; reject this invalid direct-call input before inference
        # instead of inventing zero novelty or dividing by zero.
        if not texts:
            raise ValueError("response set is empty; functional measurement requires at least one response")
        labels = partition_labels(texts, self.equivalent)
        novel = None if reference is None else novel_flags(texts, reference, self.equivalent)
        return FunctionalMetrics(
            functional_classes=len(set(labels)),
            functional_labels=labels,
            functional_novel_share=None if novel is None else sum(novel) / len(novel),
            functional_novel=novel,
            functional_window_truncated_share=(
                sum(self.exceeds_window(t) for t in texts) / len(texts) if texts else 0.0
            ),
        )

    def unload(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
