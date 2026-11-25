"""v0 diversity study: baselines for prompt-level diversity on NoveltyBench-curated.

Everything this package computes is a *diagnostic*. Distinct-n, embedding
distance, and a 1.7B rubric judge are cheap proxies for lexical variety,
semantic variety, and quality. None of them establishes that text is creative,
correct, or good. Read `DESIGN.md` before using a number in the report.
"""

# Default for configs that do not name one. Each protocol config states its own
# `protocol_version`, so running an older frozen config records the older
# version. v0.1.0 = 256-token debugging run, v0.2.0 = single seed,
# v0.3.0 = NB-curated mixed-type set, v0.4.0 = narrative scope; see DESIGN.md.
PROTOCOL_VERSION = "v0.4.0"

__all__ = ["PROTOCOL_VERSION"]
