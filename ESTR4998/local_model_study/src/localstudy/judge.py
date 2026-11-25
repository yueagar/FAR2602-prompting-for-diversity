"""Small rubric-based LLM judge.

Design choices and their reasons.

* Cross-family judge. The default judge (SmolLM2-1.7B-Instruct) is from a
  different family than the default generator (Qwen2.5-1.5B-Instruct) so that
  self-preference bias does not sit on top of the very comparison we care about.
* Absolute grading, one response at a time. Pairwise judging has a position
  bias that would need randomisation and twice the calls; absolute grading
  trades that for worse calibration, which is acceptable because we only ever
  compare arms that were graded by the same judge with the same rubric.
* Greedy decoding. The judge is measurement apparatus, so it is deterministic.
* Separate dimensions retained. Relevance, coherence and usefulness are stored
  individually; the mean is a convenience, never the only thing recorded.
* Raw text always logged. Every judge call's verbatim output is written to the
  record so a suspicious score can be audited later.

A 1.7B judge is weak. Its numbers are a screening device for the quality side
of the trade-off, not evidence about quality. The plan's human-rating study is
what calibrates them; until then, treat a quality difference under ~0.3 rubric
points as noise.

Probability-weighted scores (since 5 October 2026). Greedy decoding reports
only the judge's most likely digit, and on v0.4.0 that digit was 5 on 64.5% of
responses. G-Eval (Liu et al., EMNLP 2023) instead weights each score by the
judge's own probability of it:

    expected(dimension) = Σ_{s=1..5} s · p(s) / Σ_{s=1..5} p(s)

Here p(s) is the softmax probability of the token "s" at the position where
the greedy reply writes that dimension's score. The decoding is the same
greedy pass, so the integer scores do not change. The expectation separates
a confident 5 from a hesitant one, and a judge whose 5s are all equally
confident shows it cannot discriminate. Expected scores are stored beside the
integers (`expected_scores`, `expected_mean`), never instead of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import ModelSpec

EXPECTED_VERSION = "v0.1/g-eval-probability-weighted-digit"

# Version of the v0.1-v0.3 three-dimension judge (rubric text unchanged; v0.2
# made parsing strict). Each Rubric below carries its own version.
JUDGE_VERSION = "v0.2/absolute-3-dimension-strict-integer-1to5"

RUBRIC = """You are grading one assistant response to a user prompt.

Score three dimensions from 1 to 5 (integers only):
RELEVANCE - does it actually answer what was asked? 1 = ignores the prompt, 5 = directly answers it.
COHERENCE - is it well formed and internally consistent? 1 = incoherent or self-contradictory, 5 = clear throughout.
USEFULNESS - would the user be satisfied? 1 = useless, 5 = a good response a careful person would send.

Judge only this response. Do not reward length. Do not reward unusual wording unless it helps the user.

Reply with exactly three lines and nothing else:
RELEVANCE: <1-5>
COHERENCE: <1-5>
USEFULNESS: <1-5>"""

# Prompt-sensitivity variant: same three constructs and scale, reworded anchors
# and the dimensions asked in reverse order. It exists only to test whether the
# conclusions depend on the rubric's wording; it is never the primary judge. A
# conclusion that flips between the two is a fact about the prompt, not about
# the responses.
RUBRIC_VARIANTS = {
    "primary": RUBRIC,
    "reworded_reversed": """Rate the assistant's reply to the user's message.

Give three integer ratings on a 1-5 scale.
USEFULNESS - how much would this reply help the person who asked? 1 = not at all, 5 = exactly what they needed.
COHERENCE - is the reply clear and free of contradictions? 1 = confused or contradictory, 5 = entirely clear.
RELEVANCE - does the reply address the request? 1 = off-topic, 5 = fully on-topic.

Rate this reply on its own. Longer is not better. Unusual phrasing earns nothing by itself.

Output exactly these three lines, with no other text:
USEFULNESS: <1-5>
COHERENCE: <1-5>
RELEVANCE: <1-5>""",
}

# INITIAL_RESEARCH_SCOPE.md, "Initial operational definitions": quality is
# relevance, coherence or comprehensibility, explicit constraint satisfaction,
# and format-specific effectiveness, reported separately. For the primary domain
# the format criterion is whether the response works as a short narrative,
# character, worldbuilding element, or imaginative description. The response
# contract (one response, not a list of alternatives or an explanation of
# strategy) is part of constraint satisfaction.
SCOPE_RUBRIC = """You are grading one assistant response to a creative-writing request.

Score four dimensions from 1 to 5 (integers only):
RELEVANCE - does it respond to what was asked? 1 = ignores the request, 5 = directly fulfils it.
COHERENCE - is it comprehensible and internally consistent? 1 = incoherent or self-contradictory, 5 = clear throughout.
CONSTRAINTS - does it meet the request's explicit constraints, such as length, number of sentences, named characters, setting, or required elements, and give one response rather than a list of alternatives or an explanation of its approach? 1 = ignores them, 5 = meets every one. If the request states no explicit constraint, judge only the one-response requirement.
EFFECTIVENESS - does it work as the kind of text requested: a short narrative, character, worldbuilding element, or imaginative description? 1 = does not work as that kind of text, 5 = works well as it.

Judge only this response. Do not reward length. Do not reward unusual wording unless it makes the response work better. Invented content is appropriate when the request asks for it.

Reply with exactly four lines and nothing else:
RELEVANCE: <1-5>
COHERENCE: <1-5>
CONSTRAINTS: <1-5>
EFFECTIVENESS: <1-5>"""

SCOPE_RUBRIC_VARIANT = """Rate the assistant's reply to a creative-writing request.

Give four integer ratings on a 1-5 scale.
EFFECTIVENESS - as a story, character, invented world, or imaginative description, how well does the reply succeed? 1 = it fails as that kind of text, 5 = it succeeds fully.
CONSTRAINTS - does the reply obey every explicit requirement of the request (length, sentence count, characters, setting, required elements) and give a single piece rather than several options or commentary on its method? 1 = disregards them, 5 = obeys all. With no explicit requirement, rate only the single-piece requirement.
COHERENCE - can it be followed without confusion or contradiction? 1 = confused or contradictory, 5 = entirely clear.
RELEVANCE - does the reply do what the request asks? 1 = off-task, 5 = exactly on-task.

Rate this reply on its own. Longer is not better. Unusual phrasing earns nothing by itself. Invention is expected where the request calls for it.

Output exactly these four lines, with no other text:
EFFECTIVENESS: <1-5>
CONSTRAINTS: <1-5>
COHERENCE: <1-5>
RELEVANCE: <1-5>"""

# Retry suffixes are part of each rubric: the three-dimension one is the exact
# v0.1-v0.3 text, so those protocols still reproduce.
RETRY_SUFFIX = "\n\nYour previous reply was not in the required format. Reply with exactly the three lines."
SCOPE_RETRY_SUFFIX = "\n\nYour previous reply was not in the required format. Reply with exactly the four lines."


@dataclass(frozen=True)
class Rubric:
    """A judge prompt, the dimensions it asks for, and its rewording for checks."""

    name: str
    version: str
    text: str
    dimensions: tuple[str, ...]
    variant_text: str
    retry_suffix: str


RUBRICS = {
    "three_dim_v0_1": Rubric(
        "three_dim_v0_1", JUDGE_VERSION, RUBRIC,
        ("relevance", "coherence", "usefulness"), RUBRIC_VARIANTS["reworded_reversed"],
        RETRY_SUFFIX,
    ),
    "scope_four_part_v0_4": Rubric(
        "scope_four_part_v0_4", "v0.3/absolute-4-part-scope-strict-integer-1to5",
        SCOPE_RUBRIC, ("relevance", "coherence", "constraints", "effectiveness"),
        SCOPE_RUBRIC_VARIANT, SCOPE_RETRY_SUFFIX,
    ),
}
DEFAULT_RUBRIC = "three_dim_v0_1"

DIMENSIONS = RUBRICS[DEFAULT_RUBRIC].dimensions


def get_rubric(name: str) -> Rubric:
    if name not in RUBRICS:
        raise KeyError(f"unknown judge rubric {name!r}; known: {sorted(RUBRICS)}")
    return RUBRICS[name]


def rubric_of(quality: dict | None) -> Rubric:
    """Rubric that produced a stored quality payload; pre-v0.4 records had one."""
    return get_rubric((quality or {}).get("rubric", DEFAULT_RUBRIC))


@dataclass
class JudgeScore:
    scores: dict[str, int | None]
    parsed: bool
    attempts: int
    raw_text: str
    rubric: Rubric = RUBRICS[DEFAULT_RUBRIC]
    # Probability-weighted score per dimension (module docstring); None when
    # the judge did not compute them (fakes in tests, records before 5 Oct 2026).
    expected: dict[str, float | None] | None = None
    # New scoring calls retain every reply; absent on legacy records/fakes.
    attempt_log: list[dict] | None = None
    selected_attempt: int | None = None

    @property
    def mean(self) -> float | None:
        values = list(self.scores.values())
        if not values or any(value is None for value in values):
            return None
        return sum(values) / len(values)  # type: ignore[arg-type]

    @property
    def expected_mean(self) -> float | None:
        values = list((self.expected or {}).values())
        if not values or any(value is None for value in values):
            return None
        return sum(values) / len(values)  # type: ignore[arg-type]

    def as_dict(self) -> dict:
        payload = {
            **self.scores,
            "quality_mean": self.mean,
            "parsed": self.parsed,
            "attempts": self.attempts,
            "raw_text": self.raw_text,
            "judge_version": self.rubric.version,
            "rubric": self.rubric.name,
            "dimensions": list(self.rubric.dimensions),
        }
        if self.expected is not None:
            payload.update(expected_scores=dict(self.expected), expected_mean=self.expected_mean,
                           expected_version=EXPECTED_VERSION)
        if self.attempt_log is not None:
            payload.update(attempt_log=self.attempt_log, selected_attempt=self.selected_attempt)
        return payload


def parse_scores(
    text: str, dimensions: Sequence[str] = DIMENSIONS
) -> dict[str, int | None]:
    """Extract one integer per dimension. Missing dimensions stay None, never 0.

    A failed parse must not become a low score: that would make judge
    formatting failures look like generator quality failures.
    """
    scores: dict[str, int | None] = {}
    for dimension in dimensions:
        pattern = re.compile(
            rf"^[ \t]*{dimension}[ \t]*[:=][ \t]*([^\r\n]*)", re.IGNORECASE | re.MULTILINE
        )
        matches = pattern.findall(text)
        value = matches[0].strip() if len(matches) == 1 else ""
        scores[dimension] = int(value) if value in {"1", "2", "3", "4", "5"} else None
    return scores


def expected_from_steps(
    pieces: Sequence[str],
    digit_probs: Sequence[Sequence[float] | None],
    dimensions: Sequence[str],
) -> dict[str, float | None]:
    """Probability-weighted score per dimension from one greedy reply.

    `pieces[t]` is the text of the token generated at step t, and
    `digit_probs[t]` the probabilities of "1".."5" at that step (None where
    the token is not a digit). A dimension's score position is the digit that
    follows a line `DIMENSION:` or `DIMENSION =`, matching `parse_scores`. As
    there, a dimension written zero times or twice has no score.
    """
    result: dict[str, float | None] = {}
    prefixes = ["".join(pieces[:t]) for t in range(len(pieces))]
    for dimension in dimensions:
        label = re.compile(rf"(?:^|\n)[ \t]*{dimension}[ \t]*[:=][ \t]*$", re.IGNORECASE)
        hits = [t for t, probs in enumerate(digit_probs) if probs is not None and label.search(prefixes[t])]
        if len(hits) != 1:
            result[dimension] = None
            continue
        probs = np.asarray(digit_probs[hits[0]], dtype=float)
        total = float(probs.sum())
        result[dimension] = None if total <= 0 else float(np.dot(np.arange(1, 6), probs) / total)
    return result


def build_messages(
    prompt: str,
    response: str,
    retry: bool = False,
    rubric: str = RUBRIC,
    retry_suffix: str = RETRY_SUFFIX,
) -> list[dict[str, str]]:
    rubric = rubric + (retry_suffix if retry else "")
    user = f"[USER PROMPT]\n{prompt}\n\n[ASSISTANT RESPONSE]\n{response}"
    return [{"role": "system", "content": rubric}, {"role": "user", "content": user}]


class Judge:
    """Greedy rubric judge.

    `rubric` selects the constructs; `variant=True` uses that rubric's reworded
    text, for the prompt-sensitivity check only.

    `batch_size` responses share a forward pass. Batches are left-padded, and
    in bf16 the padding can move a score, so only `batch_size=1` makes each
    score a function of its own response alone (see `StudyConfig`).
    """

    def __init__(
        self,
        spec: ModelSpec,
        max_new_tokens: int = 96,
        batch_size: int = 1,
        rubric: Rubric | str = DEFAULT_RUBRIC,
        variant: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.spec = spec
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.rubric_spec = get_rubric(rubric) if isinstance(rubric, str) else rubric
        self.use_variant(variant)
        device = spec.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only batched generation requires left padding, otherwise the
        # shorter items in a batch continue from pad tokens and score garbage.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, revision=spec.revision, dtype=getattr(torch, spec.dtype)
        ).to(device)
        self.model.eval()
        # Token ids that spell each score "1".."5" on their own (with or without
        # a leading space), for the probability-weighted scores.
        self.digit_ids = [
            sorted({ids[0] for ids in (self.tokenizer.encode(text, add_special_tokens=False)
                                       for text in (str(score), f" {score}")) if len(ids) == 1})
            for score in range(1, 6)
        ]
        self._digit_set = {i for ids in self.digit_ids for i in ids}

    @property
    def resolved_revision(self) -> str:
        return getattr(self.model.config, "_commit_hash", None) or self.spec.revision

    def use_variant(self, variant: bool) -> None:
        """Switch between the rubric's primary and reworded text, same model."""
        self.rubric = self.rubric_spec.variant_text if variant else self.rubric_spec.text

    def _generate(
        self, message_batch: Sequence[list[dict[str, str]]]
    ) -> tuple[list[str], list[dict[str, float | None]]]:
        """Greedy replies and, from the same pass, their probability-weighted scores."""
        torch = self._torch
        rendered = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in message_batch
        ]
        encoded = self.tokenizer(
            rendered, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_logits=True,  # raw logits: the judge's own probabilities
            )
        width = encoded["input_ids"].shape[1]
        texts, expected = [], []
        for row, sequence in enumerate(output.sequences):
            generated = sequence[width:].tolist()
            texts.append(self.tokenizer.decode(sequence[width:], skip_special_tokens=True).strip())
            pieces = [self.tokenizer.decode([token]) for token in generated]
            digit_probs: list[list[float] | None] = []
            for step, token in enumerate(generated):
                if token in self._digit_set and step < len(output.logits):
                    probs = torch.softmax(output.logits[step][row].float(), dim=-1)
                    digit_probs.append([float(probs[ids].sum()) for ids in self.digit_ids])
                else:
                    digit_probs.append(None)
            expected.append(expected_from_steps(pieces, digit_probs, self.rubric_spec.dimensions))
        return texts, expected

    def score(self, prompt: str, responses: Sequence[str]) -> list[JudgeScore]:
        """Score every response to one prompt, retrying unparsable replies once."""
        def attempt(messages, text, scores, expected):
            return {"messages": messages, "raw_text": text, "scores": dict(scores),
                    "parsed": not any(value is None for value in scores.values()),
                    "expected_scores": {d: None if scores[d] is None else expected.get(d)
                                        for d in scores}}

        results: list[JudgeScore] = []
        for start in range(0, len(responses), self.batch_size):
            chunk = list(responses[start : start + self.batch_size])
            messages = [build_messages(prompt, item, rubric=self.rubric) for item in chunk]
            texts, expectations = self._generate(messages)
            for response, message, text, expected in zip(chunk, messages, texts, expectations):
                scores = parse_scores(text, self.rubric_spec.dimensions)
                history = [attempt(message, text, scores, expected)]
                selected_attempt = 1
                attempts = 1
                if any(value is None for value in scores.values()):
                    retry_message = build_messages(prompt, response, retry=True, rubric=self.rubric,
                                                   retry_suffix=self.rubric_spec.retry_suffix)
                    retry_texts, retry_expected = self._generate([retry_message])
                    attempts = 2
                    retry_scores = parse_scores(retry_texts[0], self.rubric_spec.dimensions)
                    history.append(attempt(retry_message, retry_texts[0], retry_scores, retry_expected[0]))
                    if not any(value is None for value in retry_scores.values()):
                        scores, text, expected = retry_scores, retry_texts[0], retry_expected[0]
                        selected_attempt = 2
                # A dimension the strict parser rejected has no expected score either.
                expected = {d: (None if scores[d] is None else expected.get(d)) for d in scores}
                results.append(
                    JudgeScore(
                        scores=scores,
                        parsed=not any(value is None for value in scores.values()),
                        attempts=attempts,
                        raw_text=text,
                        rubric=self.rubric_spec,
                        expected=expected,
                        attempt_log=history,
                        selected_attempt=selected_attempt,
                    )
                )
        return results

    def unload(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
