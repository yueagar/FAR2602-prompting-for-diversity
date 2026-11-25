"""Versioned research-runner foundation extracted from the second contributor's scheduling implementation.

Source: reference/tyh_2026-11-10/OpenRouter_Failure_Diagnostics_Fix/creativity_study.py
Source SHA256: fe15666649fc681db81873648eb2430abe31fd68d5bd38f001c6a8df19febb2f
Preserved route validation, conservative reservation, response sanitization and
randomized block schedule. schedule takes the validated adapter configuration;
only the original whole-protocol validators are removed (our protocol differs).
The original reference file remains unchanged. No original rubric is adopted.
"""
from decimal import Decimal
import json
import math
import random
import re

FRAMING_ALLOWANCE_TOKENS = 256
TEXT_PRICE_KEYS = frozenset({
    'prompt', 'completion', 'input_cache_read', 'input_cache_write',
    'input_cache_write_1h', 'internal_reasoning',
    'image', 'audio', 'input_audio_cache', 'web_search',
})

def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _integer(value, minimum=1):
    return type(value) is int and value >= minimum


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _price(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", value),
            "Snapshot prices must be nonnegative finite per-token decimal strings")
    amount = Decimal(value)
    require(amount.is_finite() and amount >= 0 and -100 <= amount.adjusted() <= 100,
            "Snapshot price is not a usable finite nonnegative rate")
    return amount


def text_price_tiers(pricing):
    """Validate saved rates without deleting ancillary fields or pricing tiers.

    Only min_prompt_tokens conditions are implemented. Other conditions must
    receive an explicit implementation before the route can be used.
    """
    require(isinstance(pricing, dict), 'Snapshot prices must be an object')
    base = {key: _price(pricing.get(key)) for key in ('prompt', 'completion')}
    for key, value in pricing.items():
        if key == 'discount':
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                    'Invalid endpoint discount metadata')
        elif key == 'overrides':
            continue
        elif key in TEXT_PRICE_KEYS:
            base[key] = _price(value)
        else:
            require(_price(value) == 0, 'Unsupported additional endpoint charge')
    require(base.get('input_cache_read', Decimal(0)) <= base['prompt'],
            'Cache-read price exceeds the reserved prompt rate')
    overrides = pricing.get('overrides', [])
    require(isinstance(overrides, list), 'Endpoint pricing overrides must be a list')
    tiers, thresholds = [(0, base)], set()
    for override in overrides:
        require(isinstance(override, dict)
                and set(override) <= TEXT_PRICE_KEYS | {'min_prompt_tokens'}
                and 'min_prompt_tokens' in override and len(override) > 1,
                'Unsupported or malformed endpoint pricing override')
        threshold = override['min_prompt_tokens']
        require(_integer(threshold, 0) and threshold not in thresholds,
                'Invalid or duplicate pricing threshold')
        thresholds.add(threshold)
        rates = {key: _price(value) for key, value in override.items() if key != 'min_prompt_tokens'}
        # Later matching entries override individual keys. Retain that inherited
        # state as well as earlier states, since the input count is only bounded.
        inherited = tiers[-1][1] | rates
        require(inherited.get('input_cache_read', Decimal(0)) <= inherited['prompt'],
                'Cache-read price exceeds the reserved prompt rate')
        tiers.append((threshold, inherited))
    return tiers


def _text_payload(body):
    """Enforce the request profile that makes media/search prices inapplicable."""
    from api_runner import PARAMETERS
    required = {'model', 'provider', 'messages', 'stream', 'max_tokens'}
    require(isinstance(body, dict) and required <= set(body)
            and set(body) <= required | PARAMETERS | {'response_format'},
            'Only plain-text requests without tools, plugins, media or cache directives are supported')
    require(_nonempty(body['model']) and '/' in body['model'] and ':' not in body['model'],
            'Plain-text requests require an explicit model without routing/tool modifiers')
    provider = body['provider']
    require(isinstance(provider, dict)
            and set(provider) == {'only', 'allow_fallbacks', 'require_parameters'}
            and isinstance(provider['only'], list) and len(provider['only']) == 1
            and _nonempty(provider['only'][0])
            and provider['allow_fallbacks'] is False and provider['require_parameters'] is True,
            'Plain-text requests require one pinned provider and no fallback')
    require(body['stream'] is False and _integer(body['max_tokens']),
            'Plain-text requests require nonstreaming output and a positive token cap')
    messages = body['messages']
    require(isinstance(messages, list) and messages
            and all(isinstance(message, dict) and set(message) == {'role', 'content'}
                    and message['role'] in ('system', 'user', 'assistant')
                    and isinstance(message['content'], str) for message in messages),
            'Plain-text messages cannot contain media, tools or cache directives')
    if 'response_format' in body:
        fmt = body['response_format']
        require(isinstance(fmt, dict) and set(fmt) == {'type', 'json_schema'}
                and fmt['type'] == 'json_schema', 'Only the reviewed JSON-schema response format is supported')
        schema = fmt['json_schema']
        require(isinstance(schema, dict) and set(schema) == {'name', 'strict', 'schema'}
                and _nonempty(schema['name']) and type(schema['strict']) is bool
                and isinstance(schema['schema'], dict), 'Malformed JSON-schema response format')
    if 'reasoning' in body:
        reasoning = body['reasoning']
        require(isinstance(reasoning, dict)
                and set(reasoning) <= {'effort', 'max_tokens', 'exclude', 'enabled'},
                'Unsupported reasoning fields')
        if 'max_tokens' in reasoning:
            require(_integer(reasoning['max_tokens']) and reasoning['max_tokens'] <= body['max_tokens'],
                    'Reasoning token budget must fit the total output cap')
        if 'effort' in reasoning:
            require(reasoning['effort'] in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh'),
                    'Unsupported reasoning effort')
        for key in ('exclude', 'enabled'):
            require(key not in reasoning or type(reasoning[key]) is bool, 'Invalid reasoning flag')


def validate_returned_route(raw, model, route) -> dict:
    """Pure offline check of identifiable returned identities, never of weights.

    Documented selected endpoints report provider DISPLAY NAMES, not provider.only
    variant tags. Compare them only to a saved provider_name (or literal equality
    with the pinned tag); never infer aliases or quantization. Missing/unmapped
    evidence stays unavailable, malformed/multiple selections stay ambiguous.
    Legacy top-level provider and metadata.endpoint strings may explicitly match
    the pinned tag. Any identifiable API/model/route contradiction raises
    ValueError, including contradictions alongside otherwise matching evidence.
    """
    require(isinstance(raw, dict), "Malformed API response")
    require(not raw.get("error") and not (isinstance(raw.get("choices"), list) and
            any(isinstance(c, dict) and c.get("error") for c in raw["choices"])),
            "API error; provider details suppressed")
    result = {"model_status": "unavailable", "provider_status": "unavailable",
              "endpoint_variant_status": "unavailable", "evidence": "unavailable", "sources": []}
    ambiguous = False
    if route.get('service_tier') == 'flex' and 'service_tier' in raw:
        tier = raw['service_tier']
        require(tier not in ('standard', 'default', 'priority', 'ultrafast'),
                'Returned service tier differs from saved Flex route')
        if tier != 'flex':
            ambiguous = True
    accepted_models = (model["model"], route["canonical_slug"])

    def check_model(value, source, requested=False):
        if value is not None:
            require(value == model["model"] if requested else value in accepted_models,
                    "Returned model differs from saved pinned route")
            result["sources"].append(source)
            if not requested:
                result["model_status"] = "matched"

    def check_provider(value, source, display_name=False):
        nonlocal ambiguous
        result["sources"].append(source)
        if not _nonempty(value):
            ambiguous = True
            return
        if display_name:
            expected_name = route.get("provider_name")
            if expected_name is None and value != model["provider"]:
                return  # No saved mapping: do not guess a display-name alias.
            require(value == (expected_name if expected_name is not None else model["provider"]),
                    "Returned provider differs from saved pinned route")
        else:
            require(value in (model["provider"], route.get("provider_name")),
                    "Returned endpoint tag differs from saved pinned route")
            if value == model["provider"]:
                result["endpoint_variant_status"] = "matched"
        result["provider_status"] = "matched"

    check_model(raw.get("model"), "model")
    if raw.get("provider") is not None:
        check_provider(raw["provider"], "provider")
    metadata = raw.get("openrouter_metadata")
    if metadata is not None and not isinstance(metadata, dict):
        ambiguous = True
    elif isinstance(metadata, dict):
        check_model(metadata.get("requested"), "openrouter_metadata.requested", requested=True)
        if metadata.get("endpoint") is not None:
            check_provider(metadata["endpoint"], "openrouter_metadata.endpoint")
        endpoints = metadata.get("endpoints")
        if endpoints is not None:
            available = endpoints.get("available") if isinstance(endpoints, dict) else None
            if not isinstance(available, list):
                ambiguous = True
            else:
                selected = []
                for candidate in available:
                    if not isinstance(candidate, dict) or type(candidate.get("selected")) is not bool:
                        ambiguous = True
                    elif candidate["selected"]:
                        selected.append(candidate)
                if len(selected) > 1:
                    ambiguous = True
                for candidate in selected:
                    source = "openrouter_metadata.endpoints.available"
                    check_model(candidate.get("model"), source)
                    if candidate.get("model") is None:
                        ambiguous = True
                    check_provider(candidate.get("provider"), source, display_name=True)
    result["sources"] = list(dict.fromkeys(result["sources"]))
    if ambiguous:
        result.update(provider_status="ambiguous", endpoint_variant_status="ambiguous", evidence="ambiguous")
    elif result["endpoint_variant_status"] == "matched":
        result["evidence"] = "endpoint_tag_match"
    elif result["provider_status"] == "matched":
        result["evidence"] = "provider_name_only"
    return result


def _json_safe(value):
    # Standard JSON cannot represent NaN/Infinity. Preserve the surrounding raw
    # response with an explicit marker instead of losing the whole billing record.
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            normalized = re.sub(r"[^a-z]", "", key.casefold())
            if normalized in {"authorization", "apikey", "password", "accesstoken", "secret", "headers"}:
                result[key] = "[redacted credential metadata]"
            elif normalized == "error" and item:
                code = item.get("code") if isinstance(item, dict) else None
                result[key] = {"code": code if type(code) is int else None,
                               "message": "[provider error details suppressed]"}
            else:
                result[key] = _json_safe(item)
        return result
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if type(value) is float and not math.isfinite(value):
        return "[invalid nonfinite JSON number]"
    if value is None or type(value) in (str, int, float, bool):
        return value
    return "[invalid non-JSON response value]"


def _reservation(body, route):
    # One token per UTF-8 byte is deliberately conservative for ordinary text
    # tokenizers. Include schema/metadata bytes as well, plus explicit framing.
    _text_payload(body)
    tiers = text_price_tiers(route['pricing'])
    if 'tag' in route:
        require(body['provider']['only'] == [route['tag']], 'Request provider differs from priced endpoint')
    if 'model_id' in route:
        require(body['model'] == route['model_id'], 'Request model differs from priced endpoint')
    if 'supported_parameters' in route:
        from api_runner import PARAMETERS
        required = set(body) & (PARAMETERS | {'response_format'})
        if 'response_format' in body:
            required.add('structured_outputs')
        require(required <= set(route['supported_parameters']), 'Unsupported parameters in priced payload')
    input_upper = len(canonical(body).encode("utf-8")) + FRAMING_ALLOWANCE_TOKENS
    output_cap = body["max_tokens"]
    # The true token count can be below its byte bound. Take each rate's maximum
    # over every reachable tier, including the boundary conservatively. Do not
    # use a cheaper high-input tier to underreserve smaller actual inputs.
    rates = {}
    for threshold, tier in tiers:
        if threshold <= input_upper:
            for key, price in tier.items():
                rates[key] = max(rates.get(key, Decimal(0)), price)
    # Reasoning shares the total output cap; adding its full-cap rate on top of
    # completion is deliberate overreservation, not a claim of double billing.
    # Cache-write rates are per token (including the separate one-hour rate),
    # not open-ended hourly rent. Add both as conservative padding even though
    # explicit caching is forbidden here; some endpoints create caches anyway.
    input_rate = sum((rates.get(key, Decimal(0)) for key in
                      ('prompt', 'input_cache_write', 'input_cache_write_1h')), Decimal(0))
    output_rate = rates['completion'] + rates.get('internal_reasoning', Decimal(0))
    cost = input_upper * input_rate + output_cap * output_rate
    return input_upper, cost


def schedule(cfg, tasks):
    """Randomized complete task-by-generator blocks, each containing every arm.

    Within each repetition shuffle the arm and sample orders, then rotate the arm
    order between samples to spread temporal positions. All generation precedes
    the two separate measurement types within a block. No outcomes affect order.
    """
    rng = random.Random(cfg["analysis"]["seed"])
    blocks = [{"task": t["id"], "generator": m["label"]}
              for t in tasks["tasks"] for m in cfg["models"]]
    rng.shuffle(blocks)
    for block in blocks:
        generation_order, measurement_order = [], []
        repetitions = list(range(cfg["repetitions"]))
        rng.shuffle(repetitions)
        for repetition in repetitions:
            arms = [a["id"] for a in cfg["arms"]]
            samples = list(range(cfg["samples_per_task"]))
            rng.shuffle(arms)
            rng.shuffle(samples)
            for offset, sample in enumerate(samples):
                shift = offset % len(arms)
                for arm in arms[shift:] + arms[:shift]:
                    generation_order.append({"arm": arm, "repetition": repetition, "sample": sample})
            measurement_order.extend({"arm": arm, "repetition": repetition} for arm in arms)
        rng.shuffle(measurement_order)
        block.update(generation_order=generation_order, measurement_order=measurement_order)
    return {"version": "randomized-complete-task-model-blocks-v1", "seed": cfg["analysis"]["seed"], "blocks": blocks}
