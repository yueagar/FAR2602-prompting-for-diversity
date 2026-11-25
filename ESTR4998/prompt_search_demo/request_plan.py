"""Build the adaptive search's generation and blinded evaluator requests offline.

The caller supplies the plan and completed attempts. This module constructs
request bodies and parser callbacks in memory. It does not send requests.
"""
from pathlib import Path
import sys

API = Path(__file__).resolve().parents[1] / "api_study"
sys.path.insert(0, str(API))
import api_runner as op
import api_evaluation as ev
import adaptive_demo as engine

LIVE_SCHEMA = "estr-adaptive-demo-live-v1"


def _serialized_bytes(value):
    return len(op.canonical(value).encode("utf-8"))


def sample_seed(phase, task_id, phase_task_seed, index):
    # Candidate identity is omitted for matched positions across instructions;
    # the phase is included so fresh display never reuses a selection seed.
    return int(op.fingerprint([LIVE_SCHEMA, phase, task_id, phase_task_seed, index])[:8], 16) & 0x7fffffff


def _valid_generation(attempt, cfg):
    return bool(attempt and attempt.get("status") == "ok" and attempt.get("reported_cost_usd") is not None
                and attempt.get("finish_reason") == "stop" and attempt.get("reasoning_tokens") == 0
                and type(attempt.get("reasoning_tokens")) is int and isinstance(attempt.get("text"), str)
                and attempt["text"].strip() and _serialized_bytes(attempt["text"]) <= cfg["max_response_serialized_bytes"])


def expected_jobs(plan, request, records):
    """Derive saved request identities/bodies without trusting journal metadata."""
    cfg = plan["context"]["live"]
    prefix, callback = "pd-" + request["request_id"][5:], request["request_id"]
    jobs = {}
    def add(suffix, role, body, metadata, parse=None):
        jobs[prefix + suffix] = (cfg["routes"][role], body, metadata, parse)
    if request["role"] == "proposal":
        op.require(request["round"] in (1, 2) and request == engine.proposal_request(
            plan, request["round"], request["history"], request["rejections"]), "Saved proposal binding changed")
        body = op.payload(cfg["routes"]["generator"], [{"role": "user", "content": request["prompt"]}],
            cfg["generation"] | {"seed": plan["selection_seed"] + 100000 + request["round"]})
        add("-proposal", "generator", body, {"role": "generation", "demo_role": "proposal",
            "round": request["round"], "callback_id": callback, "phase": LIVE_SCHEMA})
        return jobs
    instruction = request["instruction"]
    op.require(instruction["id"] in ("default", "naive", "proposal-r1-s1", "proposal-r1-s2", "proposal-r2-s1", "proposal-r2-s2")
               and _serialized_bytes(instruction["instruction"]) <= cfg["max_instruction_serialized_bytes"], "Saved instruction outside scope")
    ti = next(i for i, task in enumerate(plan["tasks"]) if task["id"] == request["task"]["id"])
    op.require(request["phase"] in ("selection", "fresh_display") and request == engine.evaluation_request(
        plan, instruction, ti, request["phase"]), "Saved evaluation binding changed")
    if instruction["id"] in ("default", "naive"):
        op.require(instruction == plan["reference" if instruction["id"] == "default" else "seed"], "Saved fixed control changed")
    cell = {"task": request["task"]["id"], "instruction": instruction["id"], "phase": request["phase"]}
    for i in range(5):
        body = op.payload(cfg["routes"]["generator"], [{"role": "user", "content": request["rendered_prompt"]}],
            cfg["generation"] | {"seed": sample_seed(request["phase"], request["task"]["id"], request["seed"], i)})
        add(f"-g{i}", "generator", body, {"role": "generation", "demo_role": "response", "cell": cell,
            "index": i, "callback_id": callback, "phase": LIVE_SCHEMA})
    attempts = [records.get(prefix + f"-g{i}", {}).get("attempts", []) for i in range(5)]
    if not all(rows and _valid_generation(rows[-1], cfg) for rows in attempts):
        return jobs
    texts = [rows[-1]["text"] for rows in attempts]
    order = ev.shuffle(5, request["seed"], [LIVE_SCHEMA, callback])
    add("-group", "primary", ev.judge_body(cfg["routes"]["primary"], request["task"]["text"], texts,
        "grouping", order, cfg["evaluation_profile"]), {"role": "grouping", "cell": cell,
        "order": order, "callback_id": callback, "phase": LIVE_SCHEMA}, lambda text: ev.parse_grouping(text, order))
    for i in range(5):
        add(f"-q{i}", "primary", ev.judge_body(cfg["routes"]["primary"], request["task"]["text"], [texts[i]],
            "quality", profile_id=cfg["evaluation_profile"]), {"role": "quality", "cell": cell,
            "index": i, "callback_id": callback, "phase": LIVE_SCHEMA}, ev.parse_quality)
    return jobs
