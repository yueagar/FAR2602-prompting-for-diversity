"""Prompt strategies: the things the search will eventually optimise over.

A strategy maps a benchmark prompt to the chat messages actually sent to the
generator. Baselines and (later) manually designed interventions and
search-discovered prompts all go through this one interface, so every arm is
rendered, logged, and budgeted the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

Messages = list[dict[str, str]]
StrategyFn = Callable[[str], Messages]


@dataclass(frozen=True)
class Strategy:
    key: str
    description: str
    render: StrategyFn | None  # None == registered but not implemented yet
    status: str = "implemented"


_REGISTRY: dict[str, Strategy] = {}


def register(
    key: str, description: str, status: str = "implemented"
) -> Callable[[StrategyFn], StrategyFn]:
    def decorator(function: StrategyFn) -> StrategyFn:
        if key in _REGISTRY:
            raise ValueError(f"strategy {key!r} already registered")
        _REGISTRY[key] = Strategy(key, description, function, status)
        return function

    return decorator


def register_planned(key: str, description: str) -> None:
    """Reserve a name for a strategy that is designed but not implemented.

    Keeping these in the registry means `list-strategies` shows the full planned
    comparison, and selecting one fails with a clear message instead of silently
    falling back to the default prompt.
    """
    if key in _REGISTRY:
        raise ValueError(f"strategy {key!r} already registered")
    _REGISTRY[key] = Strategy(key, description, None, status="planned")


def get(key: str) -> Strategy:
    if key not in _REGISTRY:
        raise KeyError(f"unknown strategy {key!r}; known: {sorted(_REGISTRY)}")
    strategy = _REGISTRY[key]
    if strategy.render is None:
        raise NotImplementedError(
            f"strategy {key!r} is registered as planned but not implemented "
            f"({strategy.description})"
        )
    return strategy


def render(key: str, prompt: str) -> Messages:
    return get(key).render(prompt)  # type: ignore[misc]


def all_strategies() -> list[Strategy]:
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


# --- baselines -----------------------------------------------------------
# The system prompt is held constant across every prompt-level arm so that a
# difference between arms is attributable to the arm's own text, not to one arm
# happening to have a system prompt at all.
SYSTEM_PROMPT = "You are a helpful assistant."

# v0.1.0-v0.3.0 wording. Kept so those protocols still reproduce; "diverse" is an
# odd request to make of a single response.
NAIVE_DIVERSE_NOTE = "Generate a creative and diverse answer."
# v0.4.0 wording, from INITIAL_RESEARCH_SCOPE.md ("Protocol to freeze").
NAIVE_CREATIVE_NOTE = "Generate a creative and unusual response."


@register("identity", "Benchmark prompt verbatim under a fixed system prompt.")
def _identity(prompt: str) -> Messages:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


@register(
    "naive_diverse",
    "Benchmark prompt plus a bare 'be creative and diverse' instruction. "
    "This is the straw man the search has to beat.",
)
def _naive_diverse(prompt: str) -> Messages:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{prompt}\n\n{NAIVE_DIVERSE_NOTE}"},
    ]


@register(
    "naive_creative",
    "Benchmark prompt plus a bare 'be creative and unusual' instruction "
    "(the v0.4.0 naive baseline, INITIAL_RESEARCH_SCOPE.md).",
)
def _naive_creative(prompt: str) -> Messages:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{prompt}\n\n{NAIVE_CREATIVE_NOTE}"},
    ]


# --- historical local manual-intervention placeholders ------------------
# These names record the original planned families, not implemented or frozen
# interventions. The current API manual candidates have their own protocol;
# retain this registry without treating the old October target as current.
register_planned(
    "avoidance_constraint",
    "Forbid the modal answer explicitly (e.g. 'do not give the most obvious "
    "response'). Tests whether diversity gains survive a relevance constraint.",
)
register_planned(
    "persona_shift",
    "Answer in the voice of a randomly assigned persona or discipline. Tests "
    "whether variation comes from content or only from surface style.",
)
register_planned(
    "concept_combination",
    "Require the answer to connect the prompt to a seeded unrelated concept. "
    "The seed is part of the prompt, so it is a prompt-level intervention.",
)
register_planned(
    "structured_ideation",
    "Ask for a short internal enumeration of options, then one answer drawn "
    "from it. Tests whether explicit search inside a single response helps.",
)
register_planned(
    "prior_output_conditioning",
    "Condition sample i on samples 1..i-1 and ask for something different. "
    "NOTE: this breaks the i.i.d. sampling assumption the other arms rely on, "
    "so it needs its own budget accounting and its own baseline.",
)
