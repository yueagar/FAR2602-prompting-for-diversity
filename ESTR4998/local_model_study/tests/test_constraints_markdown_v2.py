"""Opt-in Markdown sensitivity: real boundary errors and preserved legacy behavior."""
import pytest

from localstudy import constraints as frozen
from localstudy import constraints_markdown_v2 as amended


@pytest.mark.parametrize("text", [
    "**One.** Two. Three. Four. Five.",
    "*One.* Two. Three. Four. Five.",
    "__One.__ Two. Three. Four. Five.",
    "_One._ Two. Three. Four. Five.",
    "***One.*** Two. Three. Four. Five.",
    ' **"One."** Two. Three. Four. Five.',
    "One. Two. Three. Four. **Five.**",
    "**One.** **Two.** **Three.** **Four.** **Five.**",
    "**One *word*.** Two. Three. Four. Five.",
])
def test_paired_emphasis_at_both_sides_of_boundaries(text):
    assert len(amended.split_sentences(text)) == 5
    assert amended.check_prompt("curated-0", text)["sentences"]["pass"]


@pytest.mark.parametrize("text", [
    r"**One.", r"One.**", r"\**One.**", r"**One.\**",
    r"\_One._", r"_One.\_", "snake_case", "snake__case__identifier",
    "a*b*c", "2 * 3 * 4", "value.**", "** One. **", "**One.***", "***One.**",
    "`**One.**` Two.", "```**One.**```", "Inline `code` and **bold.**",
])
def test_unmatched_escaped_literal_and_code_markers_are_not_normalized(text):
    assert amended.normalize_emphasis_line(text) == text


def test_abbreviations_and_quoted_sentences_survive_emphasis_removal():
    text = '**Dr. Smith met J. K. Lane.** "Stay!" she said. It stayed? Four. Five.'
    assert amended.split_sentences(text) == [
        'Dr. Smith met J. K. Lane.', '"Stay!" she said.', 'It stayed?', 'Four.', 'Five.',
    ]


@pytest.mark.parametrize("text", [
    "", "  \n\n", "**A title**\n\n# Another heading",
    "# Heading\n\nOne. Two. Three. Four. Five.",
    "1. One.\n- Two.\n* Three.\n4) Four.\n• Five.",
    "Dr. Smith met J. K. Lane at St. Ives. She waited... then left.",
    "One.\nTwo.\n\nThree.\nFour.\nFive.",
])
def test_plain_text_heading_list_multiline_and_empty_behavior_is_preserved(text):
    assert amended.split_sentences(text) == frozen.split_sentences(text)


def test_heading_filtering_precedes_normalization():
    text = "**Title**\n\n**One.**\nTwo. Three. Four. Five."
    assert len(amended.split_sentences(text)) == 5
    assert amended.normalize_emphasis_line("**Title**") == "Title"


def test_only_sentence_count_changes_and_existing_default_stays_frozen():
    text = "**Alice.** Bob. Mallory. Four. Five."
    spec = {"sentences": 5, "paragraphs": 1, "names": ("Alice", "Bob", "Mallory"),
            "single_item": True, "lines": 1}
    original, changed = frozen.check(text, spec), amended.check(text, spec)
    assert original["sentences"]["observed"] == 4
    assert changed["sentences"]["observed"] == 5
    assert {k: v for k, v in original.items() if k != "sentences"} == {
        k: v for k, v in changed.items() if k != "sentences"}
    assert frozen.CONSTRAINTS_VERSION == amended.BASE_CONSTRAINTS_VERSION
    assert frozen.CONSTRAINTS_VERSION != amended.CONSTRAINTS_VERSION
    assert frozen.check_prompt("curated-0", text)["sentences"]["observed"] == 4


def test_unknown_constraints_still_raise_and_unannotated_prompts_remain_none():
    with pytest.raises(ValueError, match="unknown constraint"):
        amended.check("One.", {"words": 1})
    assert amended.check_prompt("curated-6", "One.") is None
    assert amended.satisfied(None) is None


def test_paragraph_and_line_counts_use_original_formatting():
    text = "**Title**\n\n**One.**\n\nSecond."
    for prompt in ("narrative-dev-02", "curated-3", "curated-10", "curated-16"):
        assert amended.check_prompt(prompt, text) == frozen.check_prompt(prompt, text)
