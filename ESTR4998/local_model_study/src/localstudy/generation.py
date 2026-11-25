"""Generator wrapper: chat rendering, seeded sampling, and budget accounting."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Sequence

from .config import DecodingSpec, ModelSpec
from .prompts import Messages


def derive_seed(base_seed: int, *parts: str) -> int:
    """Stable per-cell seed.

    Deriving from a hash of (prompt id, condition name) rather than a running
    counter means adding or reordering conditions does not change the seed of
    any existing cell, so a re-run of one arm reproduces exactly.
    """
    digest = hashlib.sha256("|".join((str(base_seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


REFERENCE_CELL = "__reference__"


def replicate_seed(base_seed: int, condition: str, prompt_id: str, replicate: int) -> int:
    """Seed for one replicate of one cell.

    Replicate 0 uses exactly the v0.2.0 derivation, so the first replicate of a
    multi-seed run is the single-seed run; replicate r > 0 adds `rep<r>` to the
    hash, so replicates never share a seed and adding one moves no other.
    """
    if replicate < 0:
        raise ValueError("replicate must be >= 0")
    if replicate == 0:
        return derive_seed(base_seed, condition, prompt_id)
    return derive_seed(base_seed, condition, prompt_id, f"rep{replicate}")


@dataclass
class GenerationResult:
    texts: list[str]
    seed: int
    prompt_tokens: int
    completion_tokens: list[int]
    truncated: list[bool]
    rendered_prompt: str
    wall_seconds: float


# Generation-config keys that change which tokens are produced but that
# DecodingSpec does not set. If a model's bundled generation_config.json sets
# any of them, generate() would apply it silently, so the generator refuses.
UNCONTROLLED_OUTPUT_KEYS = (
    "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff", "no_repeat_ngram_size",
    "encoder_repetition_penalty", "encoder_no_repeat_ngram_size", "bad_words_ids",
    "force_words_ids", "min_length", "min_new_tokens", "suppress_tokens",
    "begin_suppress_tokens", "forced_bos_token_id", "forced_eos_token_id",
    "exponential_decay_length_penalty", "sequence_bias", "renormalize_logits",
    "length_penalty", "num_beams", "penalty_alpha", "guidance_scale",
    "stop_strings", "max_time",
)


def uncontrolled_defaults(model_defaults: dict) -> dict:
    """Output-shaping settings in a model's generation config that we do not override."""
    neutral = {"num_beams": 1, "length_penalty": 1.0, "renormalize_logits": False}
    return {
        key: model_defaults[key]
        for key in UNCONTROLLED_OUTPUT_KEYS
        if model_defaults.get(key) not in (None, [], neutral.get(key, None))
    }


class Generator:
    """One pinned causal LM, sampled k times per call.

    All k samples for a cell come from a single `generate` call on a single
    prompt, so there is no padding and no cross-example batching: the set is
    reproducible from (model, revision, decoding, seed, k) alone.
    """

    def __init__(self, spec: ModelSpec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.spec = spec
        device = spec.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, revision=spec.revision, dtype=getattr(torch, spec.dtype)
        ).to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # What generate() would apply on its own; recorded, and checked for
        # anything DecodingSpec does not control.
        self.model_generation_defaults = self.model.generation_config.to_diff_dict()
        stray = uncontrolled_defaults(self.model_generation_defaults)
        if stray:
            raise ValueError(
                f"{spec.model_id} ships generation defaults the protocol does not control: "
                f"{stray}. Add them to DecodingSpec so they are explicit and recorded."
            )

    def resolved_eos_token_ids(self, decoding: DecodingSpec) -> list[int]:
        if decoding.eos_token_ids is not None:
            return list(decoding.eos_token_ids)
        eos = self.model.generation_config.eos_token_id
        return [int(eos)] if isinstance(eos, int) else [int(i) for i in eos]

    @property
    def resolved_revision(self) -> str:
        return getattr(self.model.config, "_commit_hash", None) or self.spec.revision

    def render(self, messages: Messages) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        messages: Messages,
        decoding: DecodingSpec,
        samples: int,
        seed: int,
    ) -> GenerationResult:
        torch = self._torch
        rendered = self.render(messages)
        encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_length = int(encoded["input_ids"].shape[1])

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # min_p is passed only when a config sets it, so every earlier protocol
        # calls generate() exactly as before. A model whose bundled defaults set
        # min_p is refused at load time (UNCONTROLLED_OUTPUT_KEYS).
        optional = {} if decoding.min_p is None else {"min_p": decoding.min_p}
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                do_sample=decoding.do_sample,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                top_k=decoding.top_k,
                repetition_penalty=decoding.repetition_penalty,
                eos_token_id=self.resolved_eos_token_ids(decoding),
                max_new_tokens=decoding.max_new_tokens,
                num_return_sequences=samples,
                pad_token_id=self.tokenizer.pad_token_id,
                **optional,
            )
        wall = time.perf_counter() - started

        texts: list[str] = []
        completion_tokens: list[int] = []
        truncated: list[bool] = []
        for sequence in output:
            new_tokens = sequence[prompt_length:]
            # Only pad tokens are dropped before counting. The stop token
            # (<|im_end|> for Qwen) is counted, so a response that stops by
            # itself is one token longer, in every arm alike; kept so earlier
            # runs' budget figures stay comparable. "Truncated" means the full
            # budget was used, which also includes a response that stops by
            # itself on the last allowed step.
            kept = [
                int(token)
                for token in new_tokens
                if int(token) != self.tokenizer.pad_token_id
            ]
            completion_tokens.append(len(kept))
            truncated.append(len(kept) >= decoding.max_new_tokens)
            texts.append(
                self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )
        return GenerationResult(
            texts=texts,
            seed=seed,
            prompt_tokens=prompt_length,
            completion_tokens=completion_tokens,
            truncated=truncated,
            rendered_prompt=rendered,
            wall_seconds=wall,
        )

    def unload(self) -> None:
        """Free VRAM so the judge can be loaded on the same GPU afterwards."""
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def budget_summary(results: Sequence[GenerationResult]) -> dict[str, float]:
    """Inference budget actually spent, for the matched-budget check in reports."""
    completions = [count for result in results for count in result.completion_tokens]
    return {
        "generations": len(completions),
        "completion_tokens_total": sum(completions),
        "completion_tokens_mean": sum(completions) / len(completions) if completions else 0.0,
        "prompt_tokens_total": sum(
            result.prompt_tokens * len(result.completion_tokens) for result in results
        ),
        "wall_seconds_total": sum(result.wall_seconds for result in results),
        "truncated_share": (
            sum(sum(result.truncated) for result in results) / len(completions)
            if completions
            else 0.0
        ),
    }
