"""Configuration objects for the v0 study.

One YAML file fully determines a run. Anything that could change a number is a
field here, and the whole config is hashed into every record so a table can
always be traced back to the settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from . import PROTOCOL_VERSION

COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

# Development prompts drive every decision; validation prompts only choose among
# a shortlist frozen on development; held-out prompts are touched once, after the
# method is frozen; transfer probes are a different format evaluated only after
# a shortlist exists (INITIAL_RESEARCH_SCOPE.md, data/HELDOUT_ACCESS.md).
SPLIT_NAMES = ("development", "validation", "held_out", "transfer_probe")


@dataclass(frozen=True)
class ModelSpec:
    """A Hugging Face model pinned to an exact commit.

    `revision` must be a commit sha, not a branch name: `main` moves, and a
    moving generator silently invalidates every earlier comparison.
    """

    model_id: str
    revision: str
    dtype: str = "bfloat16"
    device: str = "auto"

    def __post_init__(self) -> None:
        # A full 40-hex sha, not just "not main": any branch or tag name can
        # move, and the config fingerprint (which resume checks rely on) would
        # not change when it did (review of 7 November 2026, C04).
        if not COMMIT_SHA.fullmatch(self.revision):
            raise ValueError(
                f"{self.model_id}: pin a commit sha, not '{self.revision}'. "
                "Branch and tag names move and break comparability across runs; "
                "give the full 40-character hexadecimal commit."
            )


@dataclass(frozen=True)
class DecodingSpec:
    """Every parameter that shapes sampling or stopping, stated explicitly.

    `max_new_tokens` is part of the matched budget. Nothing is left to the
    model's bundled `generation_config.json`: Qwen2.5 ships
    `repetition_penalty: 1.1`, which `generate()` applied silently to every
    v0.1.0-v0.3.0 generation without it appearing in a config or record. The
    field now defaults to 1.1 so those runs are described exactly, and the
    generator refuses to run if the model's defaults set any other
    output-shaping parameter this spec does not control.

    `eos_token_ids` are the stopping tokens; `None` means "the model's own", and
    the resolved ids are recorded either way.

    `min_p` (Nguyen et al., ICLR 2025, "Turning Up the Heat: Min-p Sampling for
    Creative and Coherent LLM Outputs") keeps only tokens whose probability is
    at least `min_p` times the top token's. The cut-off scales with the model's
    confidence, which is what lets high temperatures stay coherent. Artificial
    Hivemind samples with min-p 0.1 at temperature 2.0, top-p 1.0. `None`
    leaves it off, as in every run before it existed.

    `top_k = 0` and `top_p = 1.0` disable those truncations. Transformers
    applies temperature before top-k, top-p and min-p, so the v0.4.0
    temperature arms sample among at most 20 tokens however high the
    temperature. Temperature-then-min-p is also the order the min-p paper
    evaluates ("After applying temperature scaling, the scaled threshold ... is
    computed"), with a base of 0.05-0.1 recommended.
    """

    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_new_tokens: int = 256
    do_sample: bool = True
    repetition_penalty: float = 1.1
    eos_token_ids: tuple[int, ...] | None = None
    min_p: float | None = None

    def __post_init__(self) -> None:
        if self.eos_token_ids is not None:
            object.__setattr__(self, "eos_token_ids", tuple(int(i) for i in self.eos_token_ids))
        if not self.do_sample:
            raise ValueError("the study samples; greedy decoding is not a protocol arm")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0 (1.0 disables it)")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0; use a greedy flag instead of 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0 (0 disables it)")
        if self.min_p is not None and not 0 < self.min_p < 1:
            raise ValueError("min_p must be in (0, 1), or None to leave it off")


@dataclass(frozen=True)
class Condition:
    """One experimental arm: a prompt strategy plus a decoding override.

    Splitting `family` from `name` keeps the report honest about what kind of
    intervention produced a number. Decoding baselines change no prompt text;
    prompt baselines change no decoding parameter.
    """

    name: str
    family: str
    strategy: str = "identity"
    decoding_overrides: dict[str, Any] = field(default_factory=dict)

    FAMILIES = (
        "decoding_baseline",
        "prompt_baseline",
        "manual_intervention",
        "searched_prompt",
    )

    def __post_init__(self) -> None:
        if self.family not in self.FAMILIES:
            raise ValueError(f"family must be one of {self.FAMILIES}, got {self.family!r}")
        unknown = set(self.decoding_overrides) - set(DecodingSpec().__dict__)
        if unknown:
            raise ValueError(f"unknown decoding overrides for {self.name}: {sorted(unknown)}")
        if self.family == "decoding_baseline" and self.strategy != "identity":
            raise ValueError(
                f"{self.name}: a decoding baseline must leave the prompt untouched "
                f"(strategy='identity'), got {self.strategy!r}"
            )
        if self.family == "prompt_baseline" and self.decoding_overrides:
            raise ValueError(
                f"{self.name}: a prompt baseline must leave decoding untouched so the "
                "prompt is the only thing that differs from `default`"
            )

    def decoding(self, base: DecodingSpec) -> DecodingSpec:
        return replace(base, **self.decoding_overrides)


# Fields added after a protocol ran -> the value that protocol's code hard-wired
# (or None for a feature that did not exist). While a config holds that value the
# field is left out of the fingerprint, so older configs keep the fingerprints
# their records carry.
LEGACY_VALUES: dict[str, Any] = {"judge_batch_size": 8, "rke_bandwidth": None}
LEGACY_DECODING_VALUES: dict[str, Any] = {"min_p": None}


@dataclass(frozen=True)
class StudyConfig:
    generator: ModelSpec
    judge: ModelSpec
    embedder: ModelSpec
    conditions: tuple[Condition, ...]
    decoding: DecodingSpec = field(default_factory=DecodingSpec)
    split: str = "development"
    # Which split file defines the prompts; relative to the study directory.
    split_file: str = "data/nb_curated_split_v0.json"
    protocol_version: str = PROTOCOL_VERSION
    samples_per_prompt: int = 10
    # Independent response sets per (condition, prompt), each with its own seed.
    # 1 reproduces protocol v0.2.0; more lets the report separate seed noise
    # from prompt-to-prompt variation.
    seed_replicates: int = 1
    # Size of the per-prompt reference set that novelty is measured against
    # (default decoding, verbatim prompt, its own seed). 0 disables novelty.
    reference_samples: int = 0
    seed: int = 20260917
    # Cosine-similarity threshold for the embedding partition proxy (see metrics).
    partition_threshold: float = 0.80
    judge_max_new_tokens: int = 96
    # Responses per judge forward pass. Batched generation left-pads, and in
    # bf16 the padding moves some scores: 6 of 40 v0.4.0 responses changed by a
    # point between batch sizes 8 and 1, so a score depended on which responses
    # shared its batch. With 1, a score depends on its own response only.
    # v0.1.0-v0.4.0 hard-wired 8, before this was a field.
    judge_batch_size: int = 1
    # Which judge rubric scores quality (judge.RUBRICS). v0.1-v0.3 used the
    # three-dimension rubric; v0.4.0 uses the scope document's four parts.
    judge_rubric: str = "three_dim_v0_1"
    # NoveltyBench v1.0 functional-equivalence classifier and its tokenizer;
    # both or neither. None skips the functional metrics.
    functional_classifier: ModelSpec | None = None
    functional_tokenizer: ModelSpec | None = None
    # Gaussian-kernel bandwidth of the RKE diversity score (metrics.rke_score),
    # fixed before a run; None skips RKE. The Vendi score needs no bandwidth.
    rke_bandwidth: float | None = None
    generation_batch_size: int = 10
    results_path: str = "results/v0_study.jsonl"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.samples_per_prompt < 2:
            raise ValueError("samples_per_prompt must be >= 2; diversity needs a set")
        if self.seed_replicates < 1:
            raise ValueError("seed_replicates must be >= 1")
        if self.reference_samples < 0:
            raise ValueError("reference_samples must be >= 0")
        if self.judge_batch_size < 1:
            raise ValueError("judge_batch_size must be >= 1")
        if self.rke_bandwidth is not None and self.rke_bandwidth <= 0:
            raise ValueError("rke_bandwidth must be > 0, or None to skip RKE")
        if (self.functional_classifier is None) != (self.functional_tokenizer is None):
            raise ValueError("functional_classifier and functional_tokenizer go together")
        from .judge import RUBRICS  # late: judge imports this module

        if self.judge_rubric not in RUBRICS:
            raise ValueError(f"unknown judge_rubric {self.judge_rubric!r}; known: {sorted(RUBRICS)}")
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")
        names = [condition.name for condition in self.conditions]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate condition names: {names}")
        if "default" not in names:
            raise ValueError("a condition named 'default' is required as the reference arm")

    # --- budget matching -------------------------------------------------
    def assert_matched_budget(self) -> None:
        """Require equal output-token ceilings within this configuration.

        Prompt set, sample count and replicate count are shared config fields.
        Declared arms may vary supported decoding parameters or prompt strategy;
        family restrictions are checked separately. This does not establish
        equal realised length, inference cost or comparability across runs.
        """
        ceilings = {
            condition.name: condition.decoding(self.decoding).max_new_tokens
            for condition in self.conditions
        }
        if len(set(ceilings.values())) > 1:
            raise ValueError(f"max_new_tokens differs across conditions: {ceilings}")

    # --- provenance ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["conditions"] = [asdict(condition) for condition in self.conditions]
        return payload

    def fingerprint(self) -> str:
        """Stable hash of the whole config; recorded on every generation.

        A field added after a protocol ran is left out while it holds the value
        that protocol's code hard-wired (`LEGACY_VALUES`), so the config that
        ran still hashes to the fingerprint its records carry.
        """
        payload = self.to_dict()
        for name, value in LEGACY_VALUES.items():
            if name in payload and payload[name] == value:
                del payload[name]
        for name, value in LEGACY_DECODING_VALUES.items():
            if name in payload["decoding"] and payload["decoding"][name] == value:
                del payload["decoding"][name]
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> StudyConfig:
    """Load a YAML config, rejecting unknown keys rather than ignoring them."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level of the config must be a mapping")

    models = {
        key: ModelSpec(**raw.pop(key)) for key in ("generator", "judge", "embedder")
    }
    for key in ("functional_classifier", "functional_tokenizer"):
        if raw.get(key) is not None:
            models[key] = ModelSpec(**raw.pop(key))
    decoding = DecodingSpec(**raw.pop("decoding", {}))
    conditions = tuple(Condition(**item) for item in raw.pop("conditions"))

    known = set(StudyConfig.__dataclass_fields__) - {"generator", "judge", "embedder",
                                                     "decoding", "conditions",
                                                     "functional_classifier",
                                                     "functional_tokenizer"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")

    config = StudyConfig(**models, decoding=decoding, conditions=conditions, **raw)
    config.assert_matched_budget()
    return config
