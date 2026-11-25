"""Opt-in, post hoc sentence-count sensitivity for paired Markdown emphasis.

This bounded heuristic is NOT a CommonMark parser or a new default. The frozen
``constraints`` module remains unchanged. Only sentence counting differs:
matched single/double/triple asterisk or underscore delimiters are removed
within an existing non-heading line before the frozen splitter is applied.
Both opening and closing delimiters matter at a sentence boundary.

Conservative limits: delimiters must be outside words, the enclosed text must
start/end without whitespace, backslash-escaped or unmatched delimiters remain
literal, and any line with an unescaped backtick is left untouched. Multiline
emphasis, HTML, links and full Markdown escaping are not interpreted. Existing
heading/list filtering and the original paragraph/name/item/line checks remain
in force. Report this profile separately; never overwrite a frozen endpoint.
"""
from __future__ import annotations

import re
from typing import Any

from . import constraints as frozen


BASE_CONSTRAINTS_VERSION = frozen.CONSTRAINTS_VERSION
CONSTRAINTS_VERSION = "v0.2/paired-emphasis-sentence-sensitivity"

# Conservative delimiter flanking deliberately excludes intraword underscores
# and asterisks. Exact equal-length delimiter runs preserve unmatched markers.
_EMPHASIS = re.compile(
    r"(?<![\w\\*_])(?P<marker>\*{1,3}|_{1,3})(?![\s*_])"
    r"(?P<body>.+?)(?<![\s\\*_])(?P=marker)(?![\w*_])"
)
_BACKTICK = re.compile(r"(?<!\\)`")


def normalize_emphasis_line(line: str) -> str:
    """Remove only recognized paired emphasis; preserve code/literal markers."""
    if _BACKTICK.search(line):
        return line
    while True:
        normalized = _EMPHASIS.sub(lambda match: match["body"], line)
        if normalized == line:
            return line
        line = normalized


def split_sentences(text: str) -> list[str]:
    """Use original heading/line rules and the opt-in emphasis normalization."""
    # Filter headings on the original text: removing a title's emphasis first
    # would otherwise turn it into an extra sentence.
    return [sentence for line in frozen._text_lines(text)
            for sentence in frozen.split_sentences(normalize_emphasis_line(line))]


def check(text: str, constraints: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Delegate every non-sentence computation to the frozen implementation."""
    result = frozen.check(text, constraints)
    if "sentences" in constraints:
        count = len(split_sentences(text))
        result["sentences"] = {"required": constraints["sentences"], "observed": count,
                               "pass": count == constraints["sentences"]}
    return result


def check_prompt(prompt_id: str, text: str) -> dict[str, dict[str, Any]] | None:
    if prompt_id not in frozen.PROMPT_CONSTRAINTS:
        return None
    return check(text, frozen.PROMPT_CONSTRAINTS[prompt_id][0])


satisfied = frozen.satisfied
