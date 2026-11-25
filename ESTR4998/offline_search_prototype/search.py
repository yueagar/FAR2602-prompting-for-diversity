"""Offline, finite-library instruction selection; no model or network client.

Counts come from supplied partitions, quality from supplied per-response scores.
Average responses, then repetitions within each task, then weight tasks equally.
Selection uses inclusive paired point-mean quality margins, NOT a statistical
non-inferiority test. Confirmation and check judging never reselect the winner.
The callback budget counts evaluator invocations, not hidden API requests.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import time

SCHEMA = "estr-offline-library-search-v1"
TOLERANCE = 1e-9


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def audit_value(value):
    """Retain malformed callback evidence as JSON without accepting its scores."""
    if type(value) is float and not math.isfinite(value):
        return {"invalid_numeric": str(value)}
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, list):
        return [audit_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: audit_value(item) for key, item in value.items()}
    return {"invalid_python_type": type(value).__name__}


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value),
            f"{name} must be finite, numeric and not boolean")
    return float(value)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def money(value):
    require(type(value) in (str, int, float), "cost must be numeric or a decimal string")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid cost") from exc
    require(result.is_finite() and result >= 0, "cost must be finite and nonnegative")
    return result


def keys(value, required, name):
    require(isinstance(value, dict) and set(value) == set(required),
            f"{name} must contain exactly {sorted(required)}")


def nonempty(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value),
            "IDs must use letters, digits, underscores or hyphens")


def unique_list(values, name):
    require(isinstance(values, list) and len(values) > 0, f"{name} must be a nonempty list")
    require(len(values) == len(set(values)), f"{name} must be unique")


def validate_plan(plan):
    """Validate a fully specified offline contract; never fill research defaults."""
    plan = json.loads(encoded(plan))
    keys(plan, {"schema", "evidence_kind", "adapter_id", "context", "tasks", "reference",
                "candidates", "selection", "confirmation", "check_tasks", "samples_per_cell",
                "quality_guards", "budget"}, "plan")
    require(plan["schema"] == SCHEMA, "wrong offline prototype schema")
    require(plan["evidence_kind"] in ("synthetic", "saved-development"), "invalid evidence kind")
    nonempty(plan["adapter_id"], "adapter_id")
    keys(plan["context"], {"generator", "primary_evaluator", "check_evaluator", "metric_version",
                            "decoding"}, "context")
    for name in ("generator", "primary_evaluator", "check_evaluator", "metric_version"):
        nonempty(plan["context"][name], name)
    require(plan["context"]["primary_evaluator"] != plan["context"]["check_evaluator"],
            "check evaluator identity must differ from primary")
    require(isinstance(plan["context"]["decoding"], dict) and plan["context"]["decoding"],
            "explicit decoding provenance required")
    integer(plan["samples_per_cell"], "samples_per_cell", 2)
    require(isinstance(plan["tasks"], list) and plan["tasks"], "tasks required")
    task_ids = []
    for task in plan["tasks"]:
        keys(task, {"id", "text", "split", "source"}, "task")
        identifier(task["id"])
        nonempty(task["text"], "task text")
        nonempty(task["source"], "task source")
        require(task["split"] == "development", "only declared development tasks are supported")
        task_ids.append(task["id"])
    unique_list(task_ids, "task IDs")
    for stage in ("selection", "confirmation"):
        keys(plan[stage], {"tasks", "seeds"}, stage)
        unique_list(plan[stage]["tasks"], f"{stage} tasks")
        require(set(plan[stage]["tasks"]) <= set(task_ids), "unknown stage task")
        unique_list(plan[stage]["seeds"], f"{stage} seeds")
        for seed in plan[stage]["seeds"]:
            integer(seed, "seed")
    require(not set(plan["selection"]["seeds"]) & set(plan["confirmation"]["seeds"]),
            "selection and confirmation seeds must be disjoint")
    unique_list(plan["check_tasks"], "check_tasks")
    require(set(plan["check_tasks"]) <= set(plan["confirmation"]["tasks"]),
            "check tasks must be confirmation tasks")
    require(isinstance(plan["candidates"], list) and plan["candidates"], "candidate library required")
    arms = [plan["reference"], *plan["candidates"]]
    for arm in arms:
        keys(arm, {"id", "instruction", "source"}, "instruction")
        identifier(arm["id"])
        require(isinstance(arm["instruction"], str), "instruction must be text")
        nonempty(arm["source"], "instruction source")
        arm["instruction"] = arm["instruction"].strip()
    unique_list([arm["id"] for arm in arms], "instruction IDs")
    # Only outer whitespace is normalized. Internal whitespace can affect prompts.
    unique_list([arm["instruction"] for arm in arms], "rendered instructions")
    require(plan["reference"]["instruction"] == "", "reference must be the unmodified task")
    require(isinstance(plan["quality_guards"], list) and plan["quality_guards"], "quality guards required")
    names = []
    for guard in plan["quality_guards"]:
        keys(guard, {"measure", "margin", "range", "eligible_tasks"}, "quality guard")
        identifier(guard["measure"])
        names.append(guard["measure"])
        require(number(guard["margin"], "margin") >= 0, "margin must be nonnegative")
        require(isinstance(guard["range"], list) and len(guard["range"]) == 2, "score range needs two bounds")
        lo, hi = [number(x, "score bound") for x in guard["range"]]
        require(lo < hi, "score range must increase")
        number(hi - lo, "score range width")
        unique_list(guard["eligible_tasks"], "eligible tasks")
        require(set(guard["eligible_tasks"]) <= set(task_ids), "unknown eligible task")
        for stage in ("selection", "confirmation"):
            require(set(guard["eligible_tasks"]) & set(plan[stage]["tasks"]),
                    f"{guard['measure']} has no eligible {stage} task")
    unique_list(names, "quality measures")
    keys(plan["budget"], {"max_evaluations", "max_cost_usd", "per_evaluation_reserve_usd"}, "budget")
    integer(plan["budget"]["max_evaluations"], "max_evaluations", 1)
    money(plan["budget"]["max_cost_usd"])
    money(plan["budget"]["per_evaluation_reserve_usd"])
    return plan


def plan_search(plan):
    """Reserve the complete library and matched winner/reference confirmation."""
    plan = validate_plan(plan)
    selection = (1 + len(plan["candidates"])) * len(plan["selection"]["tasks"]) * len(plan["selection"]["seeds"])
    confirmation = 2 * len(plan["confirmation"]["tasks"]) * len(plan["confirmation"]["seeds"])
    check = 2 * len(plan["check_tasks"]) * len(plan["confirmation"]["seeds"])
    total = selection + confirmation + check
    reserve = total * money(plan["budget"]["per_evaluation_reserve_usd"])
    require(total <= plan["budget"]["max_evaluations"], "budget cannot cover full library plus reserved confirmation/check")
    require(reserve <= money(plan["budget"]["max_cost_usd"]), "cost budget cannot cover full reservation")
    return {"selection_evaluations": selection, "confirmation_evaluations_reserved": confirmation,
            "check_evaluations_reserved": check, "maximum_evaluations": total,
            "maximum_reserved_cost_usd": str(reserve), "unit": "callback invocations, not API calls"}


def write_json(path, value):
    """Atomic publication of an internal artifact in a newly created run directory."""
    import os
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("xb") as stream:
        stream.write(encoded(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def append_event(path, value):
    import os
    with path.open("ab") as stream:
        stream.write(encoded(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def summarize(plan, cells, task_ids, seeds):
    """Mean within-task repetitions, then equal task means; unavailable is None."""
    indexed = {(cell["task_id"], cell["seed"]): cell for cell in cells}
    require(len(indexed) == len(cells), "duplicate evaluation cell")
    require(set(indexed) == {(task, seed) for task in task_ids for seed in seeds}, "incomplete or unexpected evaluation grid")
    by_task = {}
    diversity_count_sum = 0
    for task in task_ids:
        rows = [indexed[task, seed] for seed in seeds]
        task_count_sum = sum(integer(r["diversity"], "group count", 1) for r in rows)
        diversity_count_sum += task_count_sum
        by_task[task] = {"diversity": task_count_sum / len(rows), "quality": {}}
        for guard in plan["quality_guards"]:
            if task in guard["eligible_tasks"]:
                measure = guard["measure"]
                by_task[task]["quality"][measure] = math.fsum(r["quality"][measure] for r in rows) / len(rows)
    qualities = {}
    for guard in plan["quality_guards"]:
        measure = guard["measure"]
        values = [row["quality"][measure] for row in by_task.values() if measure in row["quality"]]
        qualities[measure] = {"mean": math.fsum(values) / len(values) if values else None, "tasks": len(values)}
    # Every task has the same complete seed grid. The exact integer total over
    # that common denominator preserves equal task weights without rounding
    # intermediate task means into different scores for mathematically tied arms.
    return {"diversity": diversity_count_sum / len(cells), "diversity_count_sum": diversity_count_sum,
            "quality": qualities, "by_task": by_task, "tasks": len(task_ids), "repetitions": len(seeds)}


def contrast(plan, candidate, reference):
    require(set(candidate["by_task"]) == set(reference["by_task"]), "unpaired task summaries")
    require(candidate["tasks"] == reference["tasks"] and candidate["repetitions"] == reference["repetitions"],
            "diversity contrasts require equal complete grids")
    guards = {}
    for guard in plan["quality_guards"]:
        name = guard["measure"]
        tasks = [t for t in candidate["by_task"] if t in guard["eligible_tasks"]]
        delta = (math.fsum(candidate["by_task"][t]["quality"][name] - reference["by_task"][t]["quality"][name]
                           for t in tasks) / len(tasks)) if tasks else None
        tolerance = TOLERANCE * (guard["range"][1] - guard["range"][0])
        guards[name] = {"difference": delta, "margin": guard["margin"], "tasks": len(tasks),
                        "boundary_tolerance": tolerance,
                        "point_guard_pass": delta >= -guard["margin"] - tolerance if tasks else None}
    count_difference = candidate["diversity_count_sum"] - reference["diversity_count_sum"]
    delta = count_difference / (candidate["tasks"] * candidate["repetitions"])
    decisions = [g["point_guard_pass"] for g in guards.values()]
    feasible = False if False in decisions else None if None in decisions else True
    return {"diversity_difference": delta, "diversity_count_difference": count_difference, "guards": guards,
            "estimated_feasible": feasible,
            "positive_observed_diversity_difference": count_difference > 0,
            "interpretation": "descriptive point estimates; no inferential non-inferiority verdict"}


def validate_result(plan, request, result):
    keys(result, {"request_id", "generator", "evaluator", "metric_version", "responses", "usage"}, "result")
    for key in ("request_id", "generator", "evaluator", "metric_version"):
        require(result[key] == request[key], f"result {key} differs from frozen request")
    require(isinstance(result["responses"], list) and len(result["responses"]) == plan["samples_per_cell"],
            "wrong response count")
    groups, texts, scores = [], {}, {}
    applicable = {g["measure"]: g for g in plan["quality_guards"] if request["task_id"] in g["eligible_tasks"]}
    identities = []
    for response in result["responses"]:
        keys(response, {"sample_id", "text", "group", "quality"}, "response")
        identities.append(response["sample_id"])
        nonempty(response["text"], "response text")
        group = integer(response["group"], "group")
        if response["text"] in texts:
            require(texts[response["text"]] == group, "identical responses must share a group")
        texts[response["text"]] = group
        groups.append(group)
        keys(response["quality"], set(applicable), "response quality")
        for measure, guard in applicable.items():
            value = number(response["quality"][measure], measure)
            require(guard["range"][0] <= value <= guard["range"][1], "quality score outside declared range")
            scores.setdefault(measure, []).append(value)
    require(identities == request["sample_ids"], "sample IDs/order differ from requested complete set")
    artifact = [{"sample_id": r["sample_id"], "text": r["text"]} for r in result["responses"]]
    if request["phase"] == "check":
        require(artifact == request["response_artifact"], "check judge changed the confirmed response artifact")
    return {"task_id": request["task_id"], "seed": request["seed"], "diversity": len(set(groups)),
            "quality": {name: math.fsum(values) / len(values) for name, values in scores.items()},
            "response_artifact": artifact, "response_sha256": digest(artifact)}


def run_search(plan, evaluate, output_dir, *, adapter_id):
    """Execute an offline callback experiment into a NEW directory; no resume.

    The callback is trusted offline code. This API cannot sandbox arbitrary Python
    or enforce an external account cap. Its built-in demo/replay adapters use no
    network, models, credentials or dataset loaders. Every attempted callback is
    journaled before invocation; unknown cost retains its reservation and stops.
    """
    input_plan = json.loads(encoded(plan))
    plan = validate_plan(plan)
    estimate = plan_search(plan)
    require(adapter_id == plan["adapter_id"], "adapter identity differs from plan")
    replay_source = getattr(evaluate, "replay_source_sha256", None)
    require(replay_source is None or (isinstance(replay_source, str) and
                                     re.fullmatch(r"[0-9a-f]{64}", replay_source)),
            "invalid replay source hash")
    # Execution provenance is outside the frozen evaluation request: replay
    # must request the same observations while identifying its own execution.
    execution = {"mode": "replay" if replay_source is not None else "offline_callback",
                 "source_records_sha256": replay_source,
                 "accounting_basis": "historical_replay" if replay_source is not None else "callback_reported"}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {"schema": SCHEMA, "offline_prototype": True, "input_plan": input_plan,
                "input_plan_sha256": digest(input_plan), "plan": plan, "plan_sha256": digest(plan),
                "engine_sha256": source_hash, "budget_plan": estimate, "execution": execution,
                "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "manifest.json", manifest)
    write_json(output / "config.json", plan)
    journal = output / "attempts.jsonl"
    attempts, rows = [], []
    known_cost = Decimal(0)
    unknown_reserve = Decimal(0)
    frozen = None
    winner = None
    result = {"schema": SCHEMA, "offline_prototype": True, "evidence_kind": plan["evidence_kind"],
              "status": "running", "winner": None, "selection": {}, "confirmation": None, "check": None,
              "plan_sha256": manifest["plan_sha256"], "engine_sha256": source_hash, "execution": execution,
              "limitations": ["No live adapter or protocol approval", "Point feasibility is not statistical non-inferiority",
                              "Development-task resampling is not unseen-task validation",
                              "Replay provenance cannot prove fresh remote generation", "No resume support",
                              "Adapters must enforce declared decoding/token budgets and keep judge inputs blind to arm metadata"]}

    def cell(phase, arm, task_id, seed, primary=None):
        nonlocal known_cost, unknown_reserve
        task = next(t for t in plan["tasks"] if t["id"] == task_id)
        request = {"schema": SCHEMA, "plan_sha256": manifest["plan_sha256"], "engine_sha256": source_hash,
                   "adapter_id": adapter_id, "phase": phase, "arm_id": arm["id"], "instruction": arm["instruction"],
                   "task_id": task_id, "task": task["text"], "seed": seed, "samples_per_cell": plan["samples_per_cell"],
                   "rendered_prompt": task["text"] + ("\n\n" + arm["instruction"] if arm["instruction"] else ""),
                   "generator": plan["context"]["generator"], "decoding": plan["context"]["decoding"],
                   "metric_version": plan["context"]["metric_version"],
                   "evaluator": plan["context"]["check_evaluator" if phase == "check" else "primary_evaluator"],
                   "quality_guards": [g for g in plan["quality_guards"] if task_id in g["eligible_tasks"]]}
        if primary is not None:
            request["response_artifact"] = primary["response_artifact"]
            request["sample_ids"] = [r["sample_id"] for r in primary["response_artifact"]]
        else:
            request["sample_ids"] = [f"{phase}/{arm['id']}/{task_id}/{seed}/{i}" for i in range(plan["samples_per_cell"])]
        request["request_id"] = digest(request)
        reserve = money(plan["budget"]["per_evaluation_reserve_usd"])
        require(len(attempts) < plan["budget"]["max_evaluations"], "evaluation budget exhausted")
        require(known_cost + unknown_reserve + reserve <= money(plan["budget"]["max_cost_usd"]), "cost budget exhausted")
        attempt = {"request": copy.deepcopy(request), "reserved_usd": str(reserve), "status": "pending"}
        attempts.append(attempt)
        append_event(journal, copy.deepcopy(attempt))
        unknown_reserve += reserve
        started = time.perf_counter()
        try:
            returned = evaluate(copy.deepcopy(request))
            attempt["response"] = audit_value(returned)
            # Parse cost BEFORE score validation: a malformed judgment is not free.
            usage = returned.get("usage") if isinstance(returned, dict) else None
            if (isinstance(usage, dict) and type(usage.get("underlying_requests")) is int
                    and usage["underlying_requests"] >= 0):
                attempt["reported_underlying_requests"] = usage["underlying_requests"]
            if isinstance(usage, dict) and usage.get("cost_usd") is not None:
                cost = money(usage["cost_usd"])
                known_cost += cost
                unknown_reserve -= reserve
                attempt["known_cost_usd"] = str(cost)
                require(cost <= reserve, "reported cost exceeds per-evaluation reservation")
            else:
                raise ValueError("unknown cost; reservation retained, execution halted")
            returned = json.loads(encoded(returned))
            attempt["response"] = returned
            keys(usage, {"cost_usd", "underlying_requests"}, "usage")
            integer(usage["underlying_requests"], "underlying_requests")
            evaluated = validate_result(plan, request, returned)
            attempt["status"] = "complete"
            rows.append({"request": request, "response": returned})
            return evaluated
        except Exception as exc:
            attempt["status"] = "failed"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            attempt["wall_seconds"] = time.perf_counter() - started
            append_event(journal, copy.deepcopy(attempt))

    def phase_cells(phase, arms, task_ids, seeds, originals=None):
        grid = {arm["id"]: [] for arm in arms}
        # Task/seed blocks keep every arm's evaluation grid matched.
        for task_id in task_ids:
            for seed in seeds:
                for arm in sorted(arms, key=lambda a: a["id"]):
                    prior = originals[arm["id"], task_id, seed] if originals is not None else None
                    evaluated = cell(phase, arm, task_id, seed, prior)
                    grid[arm["id"]].append(evaluated)
        return grid

    try:
        arms = [plan["reference"], *plan["candidates"]]
        stage = plan["selection"]
        grid = phase_cells("selection", arms, stage["tasks"], stage["seeds"])
        summaries = {arm: summarize(plan, cells, stage["tasks"], stage["seeds"]) for arm, cells in grid.items()}
        ref = plan["reference"]["id"]
        comparisons = {arm["id"]: contrast(plan, summaries[arm["id"]], summaries[ref]) for arm in plan["candidates"]}
        ranked = sorted((a["id"] for a in plan["candidates"] if comparisons[a["id"]]["estimated_feasible"]),
                        key=lambda name: (-summaries[name]["diversity_count_sum"], name))
        winner = ranked[0] if ranked else None
        frozen = {"winner": winner, "ranked_feasible": ranked, "summaries": summaries, "comparisons": comparisons,
                  "plan_sha256": manifest["plan_sha256"], "engine_sha256": source_hash,
                  "rule": "maximum mean group count among point-feasible candidates; exact ties by ascending ID",
                  "quality_boundary_relative_tolerance": TOLERANCE}
        write_json(output / "selection.json", frozen)
        result["selection"] = frozen
        result["winner"] = winner
        if winner is None:
            result["status"] = "no_feasible_candidate"
        else:
            chosen = next(a for a in plan["candidates"] if a["id"] == winner)
            stage = plan["confirmation"]
            paired = [plan["reference"], chosen]
            grid = phase_cells("confirmation", paired, stage["tasks"], stage["seeds"])
            summaries = {arm: summarize(plan, cells, stage["tasks"], stage["seeds"]) for arm, cells in grid.items()}
            result["confirmation"] = {"summaries": summaries, "contrast": contrast(plan, summaries[winner], summaries[ref])}
            original = {(arm, cell_["task_id"], cell_["seed"]): cell_ for arm, cells in grid.items() for cell_ in cells}
            check_grid = phase_cells("check", paired, plan["check_tasks"], stage["seeds"], original)
            checks = {arm: summarize(plan, cells, plan["check_tasks"], stage["seeds"]) for arm, cells in check_grid.items()}
            primary_subset = {arm: summarize(plan, [c for c in cells if c["task_id"] in plan["check_tasks"]],
                                            plan["check_tasks"], stage["seeds"]) for arm, cells in grid.items()}
            result["check"] = {"summaries": checks, "contrast": contrast(plan, checks[winner], checks[ref]),
                               "primary_on_same_tasks": primary_subset,
                               "primary_contrast_on_same_tasks": contrast(plan, primary_subset[winner], primary_subset[ref])}
            result["status"] = "complete"
    except Exception as exc:
        result["status"] = "incomplete"
        result["error"] = f"{type(exc).__name__}: {exc}"
        # A frozen selection survives failed confirmation; never promote runner-up.
        result["winner"] = winner if frozen is not None else None
    result["accounting"] = {"attempted_evaluations": len(attempts),
                            "completed_evaluations": sum(a["status"] == "complete" for a in attempts),
                            "known_cost_usd": str(known_cost), "unknown_reserved_usd": str(unknown_reserve),
                            "unknown_cost_attempts": sum("known_cost_usd" not in a for a in attempts),
                            "reported_underlying_requests": sum(a.get("reported_underlying_requests", 0) for a in attempts),
                            "unknown_request_usage_attempts": sum("reported_underlying_requests" not in a for a in attempts),
                            "budget_unit": "callback invocations; no external account cap enforced"}
    write_json(output / "replay.json", rows)
    write_json(output / "report.json", result)
    return result


def replay_records(path):
    """Bind exact recorded requests; absent/duplicate/changed request is an error."""
    source_bytes = Path(path).read_bytes()
    records = json.loads(source_bytes)
    require(isinstance(records, list), "replay must be a list")
    indexed = {}
    for item in records:
        keys(item, {"request", "response"}, "replay item")
        request = item["request"]
        body = {k: v for k, v in request.items() if k != "request_id"}
        require(request["request_id"] == digest(body), "replay request hash mismatch")
        require(request["request_id"] not in indexed, "duplicate replay request")
        indexed[request["request_id"]] = item

    def evaluate(request):
        require(request["request_id"] in indexed, "no exact replay record for request")
        item = indexed[request["request_id"]]
        require(item["request"] == request, "replay request differs")
        return copy.deepcopy(item["response"])

    # Hash the same bytes parsed above, not a later reread of a mutable file.
    # This identifies the supplied artifact, not authentic remote provenance.
    evaluate.replay_source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    return evaluate


def demo_plan():
    tasks = [{"id": f"toy{i}", "text": f"Invent a fictional observation for synthetic task {i}.",
              "split": "development", "source": "synthetic demonstration, not the thesis dataset"} for i in range(3)]
    arm = lambda name, instruction: {"id": name, "instruction": instruction, "source": "synthetic hand-written demonstration"}
    return {"schema": SCHEMA, "evidence_kind": "synthetic", "adapter_id": "synthetic-demo-v1",
            "context": {"generator": "synthetic-generator-v1", "primary_evaluator": "synthetic-primary-v1",
                        "check_evaluator": "synthetic-check-v1", "metric_version": "toy-partitions-v1",
                        "decoding": {"max_output_tokens": 128, "sampling": "deterministic synthetic strings"}},
            "tasks": tasks, "reference": arm("default", ""),
            "candidates": [arm("focused", "Develop one unusual causal explanation."),
                           arm("chaotic", "Mix many disconnected concepts."),
                           arm("tied", "Develop one uncommon observation.")],
            "selection": {"tasks": ["toy0", "toy1"], "seeds": [10, 11]},
            "confirmation": {"tasks": ["toy0", "toy1", "toy2"], "seeds": [20, 21]},
            "check_tasks": ["toy1"], "samples_per_cell": 4,
            "quality_guards": [{"measure": "quality", "margin": 0.5, "range": [1, 10],
                                "eligible_tasks": [t["id"] for t in tasks]},
                               {"measure": "constraint", "margin": 0.1, "range": [0, 1], "eligible_tasks": ["toy1"]}],
            "budget": {"max_evaluations": 32, "max_cost_usd": "1.00", "per_evaluation_reserve_usd": "0.01"}}


def demo_evaluator(request):
    """Deliberately fabricated scores: a flashy candidate fails quality;
    a tie is resolved by ID; the selected gain fails to persist at confirmation.
    No NLP, model, real text quality judgment, or service request occurs.
    """
    name, phase = request["arm_id"], request["phase"]
    count = {"default": 2, "focused": 3, "tied": 3, "chaotic": 4}.get(name, 2)
    if phase == "confirmation" and name == "focused":
        count = 2
    if phase == "check" and name == "focused":
        count = 1
    responses = []
    for i, sample_id in enumerate(request["sample_ids"]):
        group = i % min(count, request["samples_per_cell"])
        text = (request["response_artifact"][i]["text"] if phase == "check" else
                f"SYNTHETIC {phase} {name} {request['task_id']}: invented idea {group}.")
        quality = {g["measure"]: (4.0 if name == "chaotic" else 7.0) if g["measure"] == "quality"
                   else (0.25 if name == "chaotic" else 1.0) for g in request["quality_guards"]}
        responses.append({"sample_id": sample_id, "text": text, "group": group, "quality": quality})
    return {"request_id": request["request_id"], "generator": request["generator"],
            "evaluator": request["evaluator"], "metric_version": request["metric_version"],
            "responses": responses, "usage": {"cost_usd": "0.001", "underlying_requests": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run fabricated data only")
    demo.add_argument("--output", type=Path, required=True)
    planning = sub.add_parser("plan", help="validate an offline configuration and reserve the complete path")
    planning.add_argument("--config", type=Path, required=True)
    replay = sub.add_parser("replay", help="replay exact recorded evaluations, without generating or judging text")
    replay.add_argument("--config", type=Path, required=True)
    replay.add_argument("--records", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            plan, evaluator = demo_plan(), demo_evaluator
        else:
            plan = json.loads(args.config.read_text())
            if args.command == "plan":
                print(json.dumps(plan_search(plan), indent=2))
                return 0
            evaluator = replay_records(args.records)
        report = run_search(plan, evaluator, args.output, adapter_id=plan["adapter_id"])
        print(json.dumps({k: report[k] for k in ("status", "winner", "accounting")}, indent=2))
        return 0 if report["status"] in ("complete", "no_feasible_candidate") else 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"offline prototype: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
