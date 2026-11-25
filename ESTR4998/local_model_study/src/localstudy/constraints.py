"""Deterministic checks for the countable constraints a prompt states.

The v0.4.0 judge's CONSTRAINTS score does not track constraints that can be
counted. It gave 5 to 97% of the wrong-length answers to the five-sentence
prompts and to 99 of 127 multi-paragraph answers to "in one short paragraph"
(archive/v0_4/RESULTS.md). A count needs no judge.

IFEval (Zhou et al., 2023, "Instruction-Following Evaluation for Large Language
Models") checks such "verifiable instructions" — sentence and paragraph counts,
required words, format — with simple heuristics in code, and notes that such
heuristics produce some false negatives. This module applies the same idea to
the constraints our prompts state, with counting rules suited to them. IFEval's
own instructions name their paragraph separator (a `***` divider, or two line
breaks); ours do not, so a paragraph is a blank-line-separated block, as in
IFEval's two-line-break variant. IFEval's sentence instructions are "at least /
around / at most N"; ours ask for exactly five, counted with the splitter below.

The table below covers countable constraints on six development tasks.
Requirements that need interpretation stay with people or a stronger evaluator.

Counting rules (all deterministic; the tests pin each one):

* **Headings are not text.** A line that starts with `#`, or is wholly wrapped
  in `**`, `__` or `*` with no sentence-final punctuation, is a title. It is
  ignored by the sentence, paragraph and line counts. A title in front of a
  five-sentence story does not make it six sentences.
* **Sentences.** A sentence ends at `.`, `!` or `?`, optionally followed by
  closing quotes or brackets, then whitespace and an uppercase letter, digit
  or opening quote, or the end of the line. A line break also ends a sentence.
  A period after a common abbreviation (Mr., Dr., St., e.g. …) or a single
  initial does not end one. Leading list markers (`1.`, `-`, `*`, `•`) are
  dropped first.
  This is a rough count: across the seven splitters tried in the 5 October
  review, the number of exactly-five-sentence answers on the three
  five-sentence prompts ranged from 68 to 80 of 450. The committed rule makes
  one of those counts reproducible.
* **Paragraphs.** Blocks separated by one or more blank lines, excluding
  heading-only blocks.
* **Names.** Each name must appear as a whole word, case-sensitive.
* **Single item.** At most one line that starts with a list marker. Two or
  more list items answering "One thing…" is a list of alternatives. A list
  written as prose is not detected.
* **Lines.** Non-empty, non-heading lines, for the haiku transfer probe.
"""

from __future__ import annotations

import re
from typing import Any

CONSTRAINTS_VERSION = "v0.1/ifeval-style-sentences-paragraphs-names-single-item-lines"

# Development task ID -> (constraint specification, descriptive label).
PROMPT_CONSTRAINTS: dict[str, tuple[dict[str, Any], str]] = {
    "curated-0": ({"sentences": 5}, "sentence count"),
    "curated-2": ({"sentences": 5}, "sentence count"),
    "curated-10": ({"names": ("Alice", "Bob", "Mallory")}, "required names"),
    "curated-16": ({"single_item": True}, "single item"),
    "narrative-dev-01": ({"sentences": 5}, "sentence count"),
    "narrative-dev-02": ({"paragraphs": 1}, "paragraph count"),
}

# Words that end a sentence ("…she said no.") are deliberately absent.
ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "vs", "e.g", "i.e", "mt"})
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
_HEADING = re.compile(r"^\s*(?:#+\s+.*|(\*\*|__|\*)[^*_]+\1:?)\s*$")
# Sentence-final punctuation, optional closers, then the start of a new sentence.
_BOUNDARY = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s+[A-Z0-9\"'“‘(]|\s*$)")
_WORD = re.compile(r"\w")


def is_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(_HEADING.match(stripped)) and not re.search(r"[.!?]\W*$", stripped.strip("*_# "))


def _text_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if _WORD.search(line) and not is_heading(line)]


def split_sentences(text: str) -> list[str]:
    """Sentences of `text` under the rules in the module docstring."""
    sentences: list[str] = []
    for raw in _text_lines(text):
        line = _LIST_MARKER.sub("", raw, count=1).strip()
        start = 0
        for match in _BOUNDARY.finditer(line):
            words = line[start:match.start()].split()
            last = words[-1].lower().strip("(\"'“‘") if words else ""
            if match.group().startswith(".") and (
                last in ABBREVIATIONS or (len(last) == 1 and last.isalpha())
            ):
                continue
            piece = line[start:match.end()].strip()
            if _WORD.search(piece):
                sentences.append(piece)
            start = match.end()
        tail = line[start:].strip()
        if _WORD.search(tail):
            sentences.append(tail)
    return sentences


def count_paragraphs(text: str) -> int:
    blocks = re.split(r"\n\s*\n", text.strip())
    return sum(1 for block in blocks if _text_lines(block))


def count_list_items(text: str) -> int:
    return sum(1 for line in text.splitlines() if _LIST_MARKER.match(line) and _WORD.search(line))


def missing_names(text: str, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not re.search(rf"\b{re.escape(name)}\b", text)]


def check(text: str, constraints: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """One entry per constraint: what was required, what was observed, and pass/fail."""
    results: dict[str, dict[str, Any]] = {}
    for name, required in constraints.items():
        if name == "sentences":
            observed: Any = len(split_sentences(text))
            passed = observed == required
        elif name == "paragraphs":
            observed = count_paragraphs(text)
            passed = observed == required
        elif name == "lines":
            observed = len(_text_lines(text))
            passed = observed == required
        elif name == "names":
            observed = missing_names(text, tuple(required))
            passed = not observed
            required = list(required)
        elif name == "single_item":
            observed = count_list_items(text)
            passed = observed <= 1
        else:
            raise ValueError(f"unknown constraint {name!r}")
        results[name] = {"required": required, "observed": observed, "pass": passed}
    return results


def check_prompt(prompt_id: str, text: str) -> dict[str, dict[str, Any]] | None:
    """Checks for a response to `prompt_id`, or None when the prompt states none."""
    if prompt_id not in PROMPT_CONSTRAINTS:
        return None
    return check(text, PROMPT_CONSTRAINTS[prompt_id][0])


def satisfied(checks: dict[str, dict[str, Any]] | None) -> bool | None:
    """True when every countable constraint passes; None when there are none."""
    return None if checks is None else all(item["pass"] for item in checks.values())
