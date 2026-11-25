"""Small adaptive-search demonstration controller, with no network client.

Callbacks own model requests and hard spending controls. This controller owns
round provenance, feedback, complete-cell selection, frozen choice and replay.
Automatic scores are proxies; no inferential effectiveness claim is computed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata

SCHEMA = "estr-adaptive-prompt-demo-v1"
HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save_new(path, value, *, raw=False):
    path = Path(path)
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=raw) + "\n"
    if path.exists():
        require(path.read_text() == encoded, "Preserve existing immutable artifact: " + path.name)
        return
    with path.open("x") as stream:
        stream.write(encoded)


def save_state(path, value):
    fd, name = tempfile.mkstemp(dir=Path(path).parent, prefix="state-")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def lock(path):
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def normalized(instruction):
    return " ".join(unicodedata.normalize("NFKC", instruction).split()).casefold()


def render_prompt(task_text, instruction_text):
    return task_text if not instruction_text else task_text + "\n\nAdditional instruction:\n" + instruction_text


def validate_plan(plan):
    fields = {"schema", "tasks", "reference", "seed", "rounds", "proposals_per_round", "samples_per_cell",
              "selection_seed", "display_seed", "quality_margin", "constraint_margin", "context", "limits"}
    require(isinstance(plan, dict) and set(plan) == fields and plan["schema"] == SCHEMA, "Unexpected demo plan")
    require(plan["rounds"] == 2 and type(plan["rounds"]) is int
            and plan["proposals_per_round"] == 2 and type(plan["proposals_per_round"]) is int
            and plan["samples_per_cell"] == 5 and type(plan["samples_per_cell"]) is int,
            "This demo permits two rounds, two proposals per round and five samples per cell")
    require(plan["quality_margin"] == .5 and plan["constraint_margin"] == .05, "Demo margins differ")
    require(plan["limits"] == {"max_initial_calls": 200, "max_attempts": 400, "max_cost_usd": 5},
            "Demo limits differ from bounded scope")
    require(isinstance(plan["tasks"], list) and len(plan["tasks"]) == 2, "Exactly two development tasks required")
    ids = []
    for task in plan["tasks"]:
        require(set(task) == {"id", "text", "source", "split", "constraint_applicable"}
                and task["split"] == "development" and type(task["constraint_applicable"]) is bool,
                "Invalid task provenance or split")
        require(all(isinstance(task[k], str) and task[k].strip() for k in ("id", "text", "source"))
                and re.fullmatch(r"[a-zA-Z0-9_-]+", task["id"]), "Invalid task fields")
        ids.append(task["id"])
    require(len(set(ids)) == 2 and any(t["constraint_applicable"] for t in plan["tasks"]),
            "Duplicate tasks or no applicable constraints")
    for key, identity in (("reference", "default"), ("seed", "naive")):
        instruction = plan[key]
        require(set(instruction) == {"id", "instruction", "source"} and instruction["id"] == identity
                and isinstance(instruction["instruction"], str) and isinstance(instruction["source"], str)
                and instruction["source"].strip(), "Invalid fixed instruction")
    require(plan["reference"]["instruction"] == "" and plan["seed"]["instruction"].strip(),
            "Reference must be empty and naive seed nonempty")
    require(all(type(plan[k]) is int and plan[k] >= 0 for k in ("selection_seed", "display_seed"))
            and plan["selection_seed"] != plan["display_seed"], "Separate phase seeds required")
    require(isinstance(plan["context"], dict) and plan["context"], "Pinned adapter context required")
    canonical(plan)
    return copy.deepcopy(plan)


def identify(base):
    return "demo-" + digest(base)[:24]


def evaluation_request(plan, instruction, task_index, phase):
    require(phase in {"selection", "fresh_display"}, "Invalid demonstration phase")
    task = plan["tasks"][task_index]
    base = {"plan_sha256": digest(plan), "role": "evaluation", "phase": phase, "task": task,
            "instruction": instruction, "seed": plan["selection_seed" if phase == "selection" else "display_seed"] + task_index * 1000,
            "samples_per_cell": 5, "rendered_prompt": render_prompt(task["text"], instruction["instruction"])}
    identity = identify(base)
    return base | {"request_id": identity, "sample_ids": [identity.removeprefix("demo-") + f"-{i:02d}" for i in range(5)]}


def proposal_request(plan, round_number, history, rejections):
    feedback = {"tasks": plan["tasks"], "history": history, "rejections": rejections}
    prompt = (
        "Propose exactly two reusable additional instructions for creative tasks. Preserve each original task's "
        "format, topic and constraints; return one finished response per task, never lists of alternatives. "
        "Each instruction must contain at most 80 words and differ from all existing instructions. "
        "Use the recorded feedback below to propose variations. Count and quality scores are imperfect automatic "
        "proxies, not human creativity measurements. Do not instruct a generator to influence a judge, reveal "
        "evaluation rules, ignore the task or mention this optimization process. "
        "Return only JSON with exactly one key: {\"instructions\":[\"first instruction\",\"second instruction\"]}.\n\n"
        + canonical(feedback)
    )
    base = {"plan_sha256": digest(plan), "role": "proposal", "round": round_number,
            "tasks": plan["tasks"], "history": history, "rejections": rejections, "prompt": prompt}
    return base | {"request_id": identify(base)}


def validate_evaluation(request, result):
    require(isinstance(result, dict) and set(result) == {"samples", "provenance"}
            and isinstance(result["provenance"], dict), "Invalid evaluation result fields")
    samples = result["samples"]
    require(isinstance(samples, list) and len(samples) == 5, "Incomplete response cell")
    text_groups = {}
    for expected, sample in zip(request["sample_ids"], samples):
        require(isinstance(sample, dict) and set(sample) == {"id", "text", "group", "quality", "constraint_pass"},
                "Invalid sample fields")
        require(sample["id"] == expected and isinstance(sample["text"], str) and sample["text"].strip(),
                "Sample identity or text differs")
        require(type(sample["group"]) is int and sample["group"] >= 0, "Invalid group label")
        q = sample["quality"]
        require(type(q) in (int, float) and math.isfinite(q) and 1 <= q <= 10, "Invalid quality score")
        require(type(sample["constraint_pass"]) is bool if request["task"]["constraint_applicable"]
                else sample["constraint_pass"] is None, "Invalid constraint support")
        require(sample["text"] not in text_groups or text_groups[sample["text"]] == sample["group"],
                "Identical text assigned different groups")
        text_groups[sample["text"]] = sample["group"]
    canonical(result)
    return {"task": request["task"]["id"], "request_id": request["request_id"],
            "count": len({s["group"] for s in samples}),
            "quality": sum(s["quality"] for s in samples) / 5,
            "constraint_rate": sum(s["constraint_pass"] for s in samples) / 5
                if request["task"]["constraint_applicable"] else None}


def summarize(instruction, cells, reference, plan):
    require(len(cells) == 2 and {c["task"] for c in cells} == {t["id"] for t in plan["tasks"]},
            "Candidate needs both complete tasks")
    applicable = [c["constraint_rate"] for c in cells if c["constraint_rate"] is not None]
    summary = {"id": instruction["id"], "instruction": instruction["instruction"], "source": instruction["source"],
               "count": sum(c["count"] for c in cells) / 2,
               "count_sum": sum(c["count"] for c in cells),
               "ceiling_cells": sum(c["count"] == 5 for c in cells), "planned_cells": 2,
               "quality": sum(c["quality"] for c in cells) / 2,
               "constraint_rate": sum(applicable) / len(applicable), "cells": cells}
    if reference is None:
        summary.update(is_reference=True, estimated_feasible=None, quality_difference=None, constraint_difference=None)
    else:
        q = summary["quality"] - reference["quality"]
        constraint = summary["constraint_rate"] - reference["constraint_rate"]
        summary.update(is_reference=False, quality_difference=q, constraint_difference=constraint,
                       estimated_feasible=q >= -plan["quality_margin"] - 1e-12
                           and constraint >= -plan["constraint_margin"] - 1e-12)
    return summary


def select(candidates, incumbent=None):
    feasible = [c for c in candidates if c["estimated_feasible"] is True]
    if not feasible:
        return None, []
    best = max(c["count_sum"] for c in feasible)
    tied = sorted(c["id"] for c in feasible if c["count_sum"] == best)
    return (incumbent if incumbent in tied else tied[0]), tied


def parse_proposals(result, round_number, known):
    require(isinstance(result, dict) and set(result) == {"text", "provenance"}
            and isinstance(result["text"], str) and isinstance(result["provenance"], dict),
            "Invalid proposer callback contract")
    canonical(result)
    identities = [f"proposal-r{round_number}-s{i}" for i in (1, 2)]
    def unique_object(pairs):
        require(len({key for key, _ in pairs}) == len(pairs), "Duplicate JSON key")
        return dict(pairs)

    def reject_constant(value):
        raise ValueError("Nonfinite proposal JSON value")

    try:
        parsed = json.loads(result["text"], object_pairs_hook=unique_object, parse_constant=reject_constant)
        require(isinstance(parsed, dict) and set(parsed) == {"instructions"}
                and isinstance(parsed["instructions"], list) and len(parsed["instructions"]) == 2,
                "Malformed proposal object")
    except (ValueError, TypeError):
        return [], [{"id": identity, "reason": "malformed_round_response"} for identity in identities]
    accepted, rejected = [], []
    seen = set(known)
    for identity, raw in zip(identities, parsed["instructions"]):
        reason = None
        if not isinstance(raw, str) or not raw.strip():
            reason = "empty_or_nonstring_instruction"
        elif len(raw.split()) > 80:
            reason = "over_80_words"
        elif len(json.dumps(raw, ensure_ascii=False).encode()) > 2048:
            reason = "over_2048_serialized_utf8_bytes"
        elif normalized(raw) in seen:
            reason = "duplicate_normalized_instruction"
        if reason:
            rejected.append({"id": identity, "reason": reason, "raw_instruction": raw})
        else:
            seen.add(normalized(raw))
            accepted.append({"id": identity, "instruction": raw.strip(), "source": f"adaptive_proposer_round_{round_number}"})
    return accepted, rejected


def _trace(report):
    lines = ["# Adaptive prompt-search engineering demonstration", "",
             "Execution mode: **" + report["execution"]["mode"] + "**. " + report["execution"]["accounting_basis"], "",
             "These automatic/synthetic scores demonstrate the controller. They do not establish creativity, quality preservation or search effectiveness.", "",
             "| Candidate | Mean groups /5 | Ceiling cells /2 | Quality /10 | Constraint rate | Point-feasible |", "|---|---:|---:|---:|---:|---|"]
    for c in [report["selection"]["reference"], *report["selection"]["candidates"]]:
        lines.append(f"| {c['id']} | {c['count']:.2f} | {c['ceiling_cells']} | {c['quality']:.2f} | {c['constraint_rate']:.2f} | {c['estimated_feasible']} |")
    lines += ["", "Selected: **" + str(report["selection"]["selected_id"]) + "**.",
              "Tied best candidates: " + ", ".join(report["selection"]["tied_best_ids"]), ""]
    for r in report["rounds"]:
        lines += [f"## Round {r['round']}", "", "Proposal request: `" + r["request_id"] + "`.", ""]
        for c in r["accepted"]:
            lines += ["**" + c["id"] + "**", "", "    " + c["instruction"].replace("\n", "\n    "), ""]
        lines += ["Rejected slots: `" + canonical(r["rejected"]).replace("`", "\\`") + "`.", ""]
    lines += ["## Fresh display", "", "The choice above was frozen before these new samples. Unfavorable results never select a runner-up.", ""]
    for c in report["fresh_display"]["summaries"]:
        lines.append(f"- {c['id']}: groups {c['count']:.2f}/5, ceiling cells {c['ceiling_cells']}/2, quality {c['quality']:.2f}/10, constraints {c['constraint_rate']:.2f}.")
        lines.append("  Saved samples: " + ", ".join(
            f"[{cell['task']}](calls/{cell['request_id']}.response.json)" for cell in c["cells"]) + ".")
    lines += ["", "See `calls/` for exact requests, returned texts, feedback and provenance. A replay makes no new model calls.", ""]
    return "\n".join(lines)


def run_demo(plan, output, proposer, evaluator, *, evidence_mode, adapter_id, source_provenance=None):
    plan = validate_plan(plan)
    require(evidence_mode in {"synthetic", "live", "replay"} and isinstance(adapter_id, str) and adapter_id.strip(),
            "Explicit execution identity required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    execution = {"mode": evidence_mode, "adapter_id": adapter_id,
        "accounting_basis": {"synthetic": "Fabricated scores; zero underlying service calls or real spending.",
            "live": "Live adapter journal is authoritative for actual requests, reservations and spending.",
            "replay": "Historical replay; zero new service calls. Do not count original charges again."}[evidence_mode],
        "source_provenance": source_provenance}
    manifest = {"schema": SCHEMA, "plan": plan, "plan_sha256": digest(plan),
                "engine_sha256": file_sha(__file__), "execution": execution}
    with lock(output / ".lock"):
        save_new(output / "manifest.json", manifest)
        (output / "calls").mkdir(exist_ok=True)
        (output / "rounds").mkdir(exist_ok=True)

        def call(request, callback):
            require(file_sha(__file__) == manifest["engine_sha256"], "Engine changed during execution")
            prefix = output / "calls" / request["request_id"]
            save_new(prefix.with_suffix(".request.json"), request)
            returned = prefix.with_suffix(".response.json")
            receipt = prefix.with_suffix(".response-sha256.json")
            save_state(output / "state.json", {"status": "evaluating", "request_id": request["request_id"], "updated_utc": now()})
            if returned.exists():
                require(receipt.exists() and read(receipt) == {"request_id": request["request_id"],
                        "response_sha256": file_sha(returned)}, "Observed callback response is unbound or changed")
                return read(returned)
            result = callback(copy.deepcopy(request))
            # Retain observed callback data before validation. Invalid observed
            # data must stop and remain invalid on resume, never trigger a redraw.
            save_new(returned, result, raw=True)
            save_new(receipt, {"request_id": request["request_id"], "response_sha256": file_sha(returned)})
            return result

        def evaluate_instruction(instruction, phase, reference):
            cells = []
            for index in range(2):
                request = evaluation_request(plan, instruction, index, phase)
                cells.append(validate_evaluation(request, call(request, evaluator)))
            return summarize(instruction, cells, reference, plan)

        try:
            reference = evaluate_instruction(plan["reference"], "selection", None)
            candidates = [evaluate_instruction(plan["seed"], "selection", reference)]
            instructions = {"default": plan["reference"], "naive": plan["seed"]}
            incumbent, tied = select(candidates)
            rounds, rejected = [], []
            known = {normalized(plan["reference"]["instruction"]), normalized(plan["seed"]["instruction"])}
            for number in (1, 2):
                request = proposal_request(plan, number, [reference, *candidates], rejected)
                result = call(request, proposer)
                accepted, failures = parse_proposals(result, number, known)
                round_record = {"round": number, "request_id": request["request_id"], "accepted": accepted,
                                "rejected": failures, "feedback_sha256": digest(request["history"]),
                                "incumbent_before": incumbent}
                save_new(output / "rounds" / f"round-{number}-proposals.json", round_record)
                for instruction in accepted:
                    instructions[instruction["id"]] = instruction
                    known.add(normalized(instruction["instruction"]))
                    candidates.append(evaluate_instruction(instruction, "selection", reference))
                rejected.extend(failures)
                incumbent, tied = select(candidates, incumbent)
                round_record |= {"incumbent_after": incumbent, "tied_best_ids": tied}
                save_new(output / "rounds" / f"round-{number}-selection.json", round_record)
                rounds.append(round_record)
            selection = {"reference": reference, "candidates": candidates, "selected_id": incumbent,
                         "tied_best_ids": tied, "rejections": rejected, "effectiveness_claim": False}
            save_new(output / "selection.json", selection)
            display = {"aliases": {}, "summaries": []}
            if incumbent is not None:
                display["aliases"] = {"default": "default", "naive": "naive", "selected": incumbent}
                display_reference = evaluate_instruction(plan["reference"], "fresh_display", None)
                display["summaries"].append(display_reference)
                for identity in dict.fromkeys(("naive", incumbent)):
                    display["summaries"].append(evaluate_instruction(instructions[identity], "fresh_display", display_reference))
            report = {"schema": SCHEMA + "-report", "status": "completed" if incumbent is not None else "no_feasible_candidate",
                      "engineering_demo": True, "effectiveness_claim": False, "confirmatory": False,
                      "execution": execution, "plan_sha256": digest(plan), "rounds": rounds,
                      "selection": selection, "fresh_display": display,
                      "callback_requests": len(list((output / "calls").glob("*.request.json"))),
                      "limitations": ["Adaptive proxy selection is not validated creativity improvement.",
                          "Two familiar development tasks; fresh display is not held-out validation.",
                          "Point guards are not inferential quality-preservation tests.",
                          "No budget-matched random-search control or algorithm-superiority inference."]}
            save_new(output / "report.json", report)
            trace = output / "trace.md"
            if trace.exists():
                require(trace.read_text() == _trace(report), "Existing trace differs")
            else:
                trace.write_text(_trace(report))
            save_state(output / "state.json", {"status": report["status"], "updated_utc": now()})
            return report
        except BaseException as error:
            save_state(output / "state.json", {"status": "incomplete", "error_type": type(error).__name__, "updated_utc": now()})
            raise


def synthetic_plan():
    return {"schema": SCHEMA, "tasks": [
        {"id": "synthetic-a", "text": "Write a fictional four-sentence story about a lost key.",
         "source": "fabricated-demo-task", "split": "development", "constraint_applicable": True},
        {"id": "synthetic-b", "text": "Write a fictional four-sentence story about a late train.",
         "source": "fabricated-demo-task", "split": "development", "constraint_applicable": True}],
        "reference": {"id": "default", "instruction": "", "source": "synthetic-reference"},
        "seed": {"id": "naive", "instruction": "Be creative.", "source": "synthetic-seed"},
        "rounds": 2, "proposals_per_round": 2, "samples_per_cell": 5,
        "selection_seed": 20261119, "display_seed": 20261120, "quality_margin": .5, "constraint_margin": .05,
        "context": {"generator": "synthetic", "proposer": "synthetic-feedback-aware", "judge": "fabricated"},
        "limits": {"max_initial_calls": 200, "max_attempts": 400, "max_cost_usd": 5}}


def synthetic_proposer(request):
    if request["round"] == 1:
        strings = ["Change the cause of the central event while preserving the task.",
                   "Use disconnected surprises regardless of coherence."]
    else:
        require(any(h["id"] == "proposal-r1-s1" and h["count"] == 4 for h in request["history"]),
                "Round two requires actual round-one feedback")
        strings = ["Build on the promising causal variation by changing who understands its consequences.",
                   "  Change the cause of the central event while preserving the task.  "]
    return {"text": json.dumps({"instructions": strings}), "provenance": {"synthetic": True, "underlying_requests": 0}}


def synthetic_evaluator(request):
    identity = request["instruction"]["id"]
    count = {"default": 2, "naive": 3, "proposal-r1-s1": 4, "proposal-r1-s2": 5, "proposal-r2-s1": 5}.get(identity, 4)
    quality = 3 if identity == "proposal-r1-s2" else 8
    if request["phase"] == "fresh_display" and identity == "proposal-r2-s1":
        count, quality = 2, 7.2
    return {"samples": [{"id": sample, "text": "Fabricated response " + sample, "group": i % count,
                         "quality": quality, "constraint_pass": True}
                        for i, sample in enumerate(request["sample_ids"])],
            "provenance": {"synthetic": True, "underlying_requests": 0, "actual_cost_usd": 0}}


def replay_callbacks(source):
    source = Path(source).resolve()
    manifest = read(source / "manifest.json")
    require(manifest["engine_sha256"] == file_sha(__file__), "Replay requires the original engine source")
    inventory = {str(p.relative_to(source)): file_sha(p) for p in sorted((source / "calls").glob("*.json"))}

    def callback(request):
        prefix = source / "calls" / request["request_id"]
        require(read(prefix.with_suffix(".request.json")) == request, "Replay request mismatch")
        response = prefix.with_suffix(".response.json")
        require(response.exists(), "Missing replay response")
        require(all(file_sha(source / name) == h for name, h in inventory.items()), "Replay source changed")
        return read(response)
    return manifest["plan"], callback, {"source": str(source), "manifest_sha256": file_sha(source / "manifest.json"),
                                       "records_sha256": inventory}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "replay"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args(argv)
    if args.command == "demo":
        report = run_demo(synthetic_plan(), args.output, synthetic_proposer, synthetic_evaluator,
                          evidence_mode="synthetic", adapter_id="synthetic-adaptive-v1")
    else:
        require(args.source is not None and args.source.resolve() != args.output.resolve(), "Distinct replay source required")
        plan, callback, provenance = replay_callbacks(args.source)
        report = run_demo(plan, args.output, callback, callback, evidence_mode="replay", adapter_id="exact-record-replay-v1",
                          source_provenance=provenance)
    print(json.dumps({"status": report["status"], "mode": report["execution"]["mode"],
                      "selected": report["selection"]["selected_id"], "callback_requests": report["callback_requests"],
                      "effectiveness_claim": False}))


if __name__ == "__main__":
    main()
