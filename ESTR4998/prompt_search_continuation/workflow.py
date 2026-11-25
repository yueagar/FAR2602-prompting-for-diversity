"""Post-observation continuation of one stopped prompt-search demonstration.

No network client: the runtime owns transport, accounting and strict certification
of length-only failures. Original callbacks and proposal histories remain frozen.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import tempfile

BASE_PATH = Path(__file__).resolve().parents[1] / "prompt_search_demo" / "adaptive_demo.py"
_spec = importlib.util.spec_from_file_location("continuation_original_engine", BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
SCHEMA = "estr-prompt-search-continuation-v1"


class CellTruncated(Exception):
    """Only a runtime-certified, observed, known-cost length failure is skippable."""

    def __init__(self, request_id, api_request_id, receipt):
        self.request_id = request_id
        self.api_request_id = api_request_id
        self.receipt = copy.deepcopy(receipt)
        super().__init__("Observed length failure: " + api_request_id)


def validate_truncation(request, receipt):
    fields = {"schema", "request_id", "api_request_id", "request_file_sha256", "attempt",
              "finish_reason", "reasoning_tokens", "known_cost_usd", "observed_text_nonempty",
              "other_controls_passed"}
    base.require(isinstance(receipt, dict) and set(receipt) == fields
                 and receipt["schema"] == "estr-continuation-length-v1", "Invalid length receipt")
    prefix = request["request_id"].replace("demo-", "pd-")
    base.require(request["role"] == "evaluation" and receipt["request_id"] == request["request_id"]
                 and receipt["api_request_id"] in {prefix + f"-g{i}" for i in range(5)},
                 "Length receipt request differs")
    base.require(isinstance(receipt["request_file_sha256"], str)
                 and re.fullmatch(r"[0-9a-f]{64}", receipt["request_file_sha256"])
                 and type(receipt["attempt"]) is int and 0 <= receipt["attempt"] <= 1,
                 "Invalid length receipt provenance")
    cost = receipt["known_cost_usd"]
    base.require(type(cost) in (int, float) and math.isfinite(cost) and cost >= 0
                 and receipt["finish_reason"] == "length"
                 and type(receipt["reasoning_tokens"]) is int and receipt["reasoning_tokens"] == 0
                 and receipt["observed_text_nonempty"] is True
                 and receipt["other_controls_passed"] is True,
                 "Only known-cost length with all other controls passed is skippable")
    return copy.deepcopy(receipt)


def _bound_response(calls, request):
    prefix = calls / request["request_id"]
    requested = prefix.with_suffix(".request.json")
    returned = prefix.with_suffix(".response.json")
    receipt = prefix.with_suffix(".response-sha256.json")
    base.require(requested.exists() and base.read(requested) == request, "Saved request differs")
    base.require(returned.exists() and receipt.exists() and base.read(receipt) == {
        "request_id": request["request_id"], "response_sha256": base.file_sha(returned)},
        "Observed callback response is unbound or changed")
    return base.read(returned)


def _snapshot_parent(plan, parent_trace, amendment):
    base.require(amendment.get("schema") == SCHEMA, "Unexpected amendment schema")
    parent = Path(amendment["parent"]["path"]).resolve()
    base.require(parent_trace == parent / "trace", "Parent trace differs from amendment")
    for name, sha in amendment["parent"]["files_sha256"].items():
        target = (parent / name).resolve()
        base.require(target.is_relative_to(parent) and target.is_file()
                     and base.file_sha(target) == sha, "Frozen parent evidence differs: " + name)
    plan_path = Path(amendment["original_plan"]["path"])
    base.require(base.file_sha(plan_path) == amendment["original_plan"]["sha256"]
                 and base.read(plan_path) == plan, "Original plan differs")
    manifest = base.read(parent_trace / "manifest.json")
    base.require(manifest["plan"] == plan and manifest["plan_sha256"] == base.digest(plan)
                 and manifest["engine_sha256"] == base.file_sha(BASE_PATH), "Original engine or plan changed")
    required = {"trace/manifest.json", "trace/state.json"}
    required.update(str(p.relative_to(parent)) for p in (parent_trace / "calls").glob("*.json"))
    required.update(str(p.relative_to(parent)) for p in (parent_trace / "rounds").glob("*.json"))
    raw = "api/requests/" + amendment["parent_truncation"]["api_request_id"] + ".json"
    required.add(raw)
    base.require(required.issubset(amendment["parent"]["files_sha256"]), "Parent binding omits required evidence")
    base.require(base.file_sha(parent / raw) == amendment["parent_truncation"]["request_file_sha256"],
                 "Original truncation raw-file binding differs")


def _parent_history(plan, parent_trace, amendment):
    """Reconstruct the exact original pre-stop history without model callbacks."""
    _snapshot_parent(plan, parent_trace, amendment)
    calls = parent_trace / "calls"
    seen = set()
    results = {}

    def observed(request):
        result = _bound_response(calls, request)
        seen.add(request["request_id"])
        results[request["request_id"]] = (request, result)
        return result

    def evaluate(instruction, reference):
        cells = []
        for index in range(2):
            request = base.evaluation_request(plan, instruction, index, "selection")
            cells.append(base.validate_evaluation(request, observed(request)))
        return base.summarize(instruction, cells, reference, plan)

    reference = evaluate(plan["reference"], None)
    candidates = [evaluate(plan["seed"], reference)]
    instructions = {"default": plan["reference"], "naive": plan["seed"]}
    incumbent, _ = base.select(candidates)
    known = {base.normalized(i["instruction"]) for i in instructions.values()}
    rejections, rounds = [], []
    for number in (1, 2):
        request = base.proposal_request(plan, number, [reference, *candidates], rejections)
        accepted, rejected = base.parse_proposals(observed(request), number, known)
        base.require([a["id"] for a in accepted] == [f"proposal-r{number}-s1", f"proposal-r{number}-s2"]
                     and not rejected, "Amendment requires the original four accepted proposals")
        record = {"round": number, "request_id": request["request_id"], "accepted": accepted,
                  "rejected": rejected, "feedback_sha256": base.digest(request["history"]),
                  "incumbent_before": incumbent}
        base.require(base.read(parent_trace / "rounds" / f"round-{number}-proposals.json") == record,
                     "Original proposal record or feedback differs")
        for instruction in accepted:
            instructions[instruction["id"]] = instruction
            known.add(base.normalized(instruction["instruction"]))
            if number == 1:
                candidates.append(evaluate(instruction, reference))
        if number == 1:
            incumbent, tied = base.select(candidates, incumbent)
            record |= {"incumbent_after": incumbent, "tied_best_ids": tied}
            base.require(base.read(parent_trace / "rounds/round-1-selection.json") == record,
                         "Original round-one selection differs")
        rounds.append(record)
    truncated_instruction = instructions["proposal-r2-s1"]
    first = base.evaluation_request(plan, truncated_instruction, 0, "selection")
    retained_cell = base.validate_evaluation(first, observed(first))
    interrupted = base.evaluation_request(plan, truncated_instruction, 1, "selection")
    prefix = calls / interrupted["request_id"]
    base.require(base.read(prefix.with_suffix(".request.json")) == interrupted
                 and not prefix.with_suffix(".response.json").exists(), "Original interrupted cell differs")
    truncation = validate_truncation(interrupted, amendment["parent_truncation"])
    base.require({p.name.removesuffix(".request.json") for p in calls.glob("*.request.json")}
                 == seen | {interrupted["request_id"]}, "Unexpected parent callback request")
    base.require({p.name.removesuffix(".response.json") for p in calls.glob("*.response.json")} == seen,
                 "Unexpected parent callback response")
    base.require(not (parent_trace / "selection.json").exists()
                 and not (parent_trace / "rounds/round-2-selection.json").exists(),
                 "Parent already selected after round two")
    exclusion = {"id": truncated_instruction["id"], "reason": "observed_generation_length",
                 "whole_candidate_excluded": True, "retained_complete_cells": [retained_cell],
                 "interrupted_request": interrupted, "truncation": truncation}
    return {"reference": reference, "candidates": candidates, "instructions": instructions,
            "incumbent": incumbent, "rounds": rounds, "rejections": rejections,
            "observed": results, "exclusion": exclusion}


def _trace(report):
    lines = ["# Amended prompt-search continuation", "",
             "A post-observation engineering amendment; the original plan remains incomplete.", "",
             "Execution: **" + report["execution"]["mode"] + "**. " + report["execution"]["accounting_basis"], "",
             "Workflow complete: **" + str(report["workflow_complete"]) + "**. Fresh display complete: **"
             + str(report["fresh_display_complete"]) + "**.", "",
             "Automatic point scores do not establish creativity improvement or search effectiveness.", "",
             "| Complete selection candidate | Mean groups /5 | Ceiling cells /2 | Quality /10 | Constraint rate | Point-feasible |",
             "|---|---:|---:|---:|---:|---|"]
    selection = report["selection"]
    for c in [selection["reference"], *selection["candidates"]]:
        lines.append(f"| {c['id']} | {c['count']:.2f} | {c['ceiling_cells']} | {c['quality']:.2f} | {c['constraint_rate']:.2f} | {c['estimated_feasible']} |")
    lines += ["", "Frozen amended selection: **" + str(selection["selected_id"]) + "**.",
              "Tied best: " + ", ".join(selection["tied_best_ids"]) + ".", "", "## Exclusions", ""]
    for exclusion in selection["exclusions"]:
        lines += ["- **" + exclusion["id"] + "**: whole candidate excluded after observed length; retained complete cells: "
                  + str(len(exclusion["retained_complete_cells"])) + ". No aggregate or rank is assigned.",
                  "  Raw request: `" + exclusion["truncation"]["api_request_id"] + "`."]
    lines += ["", "## Fresh display", "", "The selection above preceded fresh display and never changes in response to its outcomes.", "",
              "| Instruction | Task | Status | Groups /5 | Quality /10 | Constraint rate | Saved evidence |", "|---|---|---|---:|---:|---:|---|"]
    for item in report["fresh_display"]["cells"]:
        c = item.get("summary")
        if c is None:
            link = f"[length receipt](../trace/calls/{item['request_id']}.truncation.json)"
            lines.append(f"| {item['instruction_id']} | {item['task']} | missing: observed length | — | — | — | {link} |")
        else:
            constraint = "—" if c["constraint_rate"] is None else f"{c['constraint_rate']:.2f}"
            link = f"[five responses](../trace/calls/{item['request_id']}.response.json)"
            lines.append(f"| {item['instruction_id']} | {item['task']} | complete (n=5) | {c['count']} | {c['quality']:.2f} | {constraint} | {link} |")
    lines += ["", "Complete fresh-display instruction aggregates:", ""]
    for c in report["fresh_display"]["summaries"]:
        lines.append(f"- {c['id']}: groups {c['count']:.2f}/5; ceiling cells {c['ceiling_cells']}/2; quality {c['quality']:.2f}/10; constraints {c['constraint_rate']:.2f}.")
    if not report["fresh_display"]["summaries"]:
        lines.append("- None.")
    lines += ["", "Missing cells are never replaced or scored as zero. An incomplete default display prevents between-instruction display comparisons.",
              "The same two familiar tasks and imperfect proxies remain; no held-out or inferential efficacy claim is made.",
              "Exact original proposals/history and callback records remain saved; this continuation makes no additional proposal calls.", ""]
    return "\n".join(lines)


def run_continuation(plan, parent_trace, output, evaluator, *, amendment,
                     evidence_mode="live", adapter_id="continuation-runtime-v1", source_provenance=None):
    plan = base.validate_plan(plan)
    parent_trace = Path(parent_trace).resolve()
    output = Path(output).resolve()
    base.require(evidence_mode in {"live", "synthetic", "replay"}, "Explicit execution mode required")
    base.require(output != parent_trace.parent and not output.is_relative_to(parent_trace.parent),
                 "Continuation must not write within parent evidence")
    parent = _parent_history(plan, parent_trace, amendment)
    output.mkdir(parents=True, exist_ok=True)
    work = output / "continuation"
    calls = output / "trace/calls"
    work.mkdir(exist_ok=True)
    calls.mkdir(parents=True, exist_ok=True)
    execution = {"mode": evidence_mode, "adapter_id": adapter_id, "source_provenance": source_provenance,
                 "accounting_basis": {"live": "Runtime journal includes inherited charges once and new calls cumulatively.",
                     "synthetic": "New callbacks are fabricated; zero new service calls or spending. Inherited parent observations and costs remain historical.",
                     "replay": "Historical exact replay; zero new service calls or spending."}[evidence_mode]}
    manifest = {"schema": SCHEMA, "amendment": amendment, "amendment_sha256": base.digest(amendment),
                "original_plan": plan, "original_plan_sha256": base.digest(plan), "parent_trace": str(parent_trace),
                "workflow_sha256": base.file_sha(__file__), "original_engine_sha256": base.file_sha(BASE_PATH),
                "execution": execution}
    with base.lock(work / ".lock"):
        base.save_new(work / "manifest.json", manifest)

        def call(request):
            base.require(base.file_sha(__file__) == manifest["workflow_sha256"]
                         and base.file_sha(BASE_PATH) == manifest["original_engine_sha256"], "Workflow source changed")
            prefix = calls / request["request_id"]
            base.save_new(prefix.with_suffix(".request.json"), request)
            returned = prefix.with_suffix(".response.json")
            receipt_path = prefix.with_suffix(".response-sha256.json")
            truncated = prefix.with_suffix(".truncation.json")
            truncation_hash = prefix.with_suffix(".truncation-sha256.json")
            base.require(not (returned.exists() and truncated.exists()), "Conflicting observed outcomes")
            if returned.exists():
                return _bound_response(calls, request)
            if truncated.exists():
                base.require(truncation_hash.exists() and base.read(truncation_hash) == {
                    "request_id": request["request_id"], "truncation_sha256": base.file_sha(truncated)},
                    "Length receipt is unbound or changed")
                receipt = validate_truncation(request, base.read(truncated))
                raise CellTruncated(request["request_id"], receipt["api_request_id"], receipt)
            base.require(not receipt_path.exists() and not truncation_hash.exists(), "Orphan outcome binding")
            base.save_state(work / "state.json", {"status": "evaluating", "request_id": request["request_id"]})
            try:
                result = evaluator(copy.deepcopy(request))
            except CellTruncated as error:
                base.require(error.request_id == request["request_id"]
                             and error.api_request_id == error.receipt.get("api_request_id"), "Truncation exception differs")
                receipt = validate_truncation(request, error.receipt)
                base.save_new(truncated, receipt)
                base.save_new(truncation_hash, {"request_id": request["request_id"],
                                              "truncation_sha256": base.file_sha(truncated)})
                raise
            base.save_new(returned, result, raw=True)
            base.save_new(receipt_path, {"request_id": request["request_id"], "response_sha256": base.file_sha(returned)})
            return result

        try:
            # Verbatim copies preserve every parent observed callback and its hash.
            for identity in parent["observed"]:
                for suffix in ("request.json", "response.json", "response-sha256.json"):
                    name = identity + "." + suffix
                    source = (parent_trace / "calls" / name).read_bytes()
                    target = calls / name
                    if target.exists():
                        base.require(target.read_bytes() == source, "Inherited callback changed")
                    else:
                        with target.open("xb") as stream:
                            stream.write(source)
            interrupted = parent["exclusion"]["interrupted_request"]
            source = parent_trace / "calls" / (interrupted["request_id"] + ".request.json")
            target = calls / source.name
            if target.exists():
                base.require(target.read_bytes() == source.read_bytes(), "Inherited interrupted request changed")
            else:
                with target.open("xb") as stream:
                    stream.write(source.read_bytes())
            reference = parent["reference"]
            candidates = copy.deepcopy(parent["candidates"])
            exclusions = [copy.deepcopy(parent["exclusion"])]
            instruction = parent["instructions"]["proposal-r2-s2"]
            cells = []
            for index in range(2):
                request = base.evaluation_request(plan, instruction, index, "selection")
                try:
                    cells.append(base.validate_evaluation(request, call(request)))
                except CellTruncated as error:
                    exclusions.append({"id": instruction["id"], "reason": "observed_generation_length",
                        "whole_candidate_excluded": True, "retained_complete_cells": cells,
                        "interrupted_request": request, "truncation": error.receipt})
                    break
            else:
                candidates.append(base.summarize(instruction, cells, reference, plan))
            selected, tied = base.select(candidates, parent["incumbent"])
            selection = {"schema": SCHEMA + "-selection", "original_plan_sha256": base.digest(plan),
                         "amendment_sha256": base.digest(amendment), "reference": reference,
                         "candidates": candidates, "selected_id": selected, "tied_best_ids": tied,
                         "incumbent_before": parent["incumbent"], "exclusions": exclusions,
                         "rejections": parent["rejections"], "original_plan_complete": False, "effectiveness_claim": False}
            base.save_new(work / "selection.json", selection)
            display = {"aliases": {}, "cells": [], "summaries": [], "comparison_available": False}
            if selected is not None:
                display["aliases"] = {"default": "default", "naive": "naive", "selected": selected}
                complete = {}
                for identity in dict.fromkeys(("default", "naive", selected)):
                    instruction = parent["instructions"][identity]
                    cells = []
                    for index in range(2):
                        request = base.evaluation_request(plan, instruction, index, "fresh_display")
                        item = {"instruction_id": identity, "task": request["task"]["id"], "request_id": request["request_id"]}
                        try:
                            summary = base.validate_evaluation(request, call(request))
                            cells.append(summary)
                            item.update(status="complete", summary=summary)
                        except CellTruncated as error:
                            item.update(status="missing_observed_length", summary=None, truncation=error.receipt)
                        display["cells"].append(item)
                    if len(cells) == 2:
                        complete[identity] = cells
                display_reference = (base.summarize(plan["reference"], complete["default"], None, plan)
                                     if "default" in complete else None)
                display["comparison_available"] = display_reference is not None
                for identity, cells in complete.items():
                    summary = base.summarize(parent["instructions"][identity], cells,
                                             None if identity == "default" else display_reference, plan)
                    if identity != "default" and display_reference is None:
                        summary.update(is_reference=False, estimated_feasible=None,
                                       quality_difference=None, constraint_difference=None)
                    display["summaries"].append(summary)
            display_complete = bool(display["aliases"]) and all(c["status"] == "complete" for c in display["cells"])
            status = ("workflow_complete_no_feasible_candidate" if selected is None else
                      "workflow_complete" if display_complete else "workflow_complete_with_missing_display")
            report = {"schema": SCHEMA + "-report", "status": status, "workflow_complete": True,
                      "fresh_display_complete": display_complete, "fresh_display_required": selected is not None,
                      "original_plan_complete": False, "engineering_demo": True, "effectiveness_claim": False,
                      "confirmatory": False, "post_observation_amendment": True, "execution": execution,
                      "amendment_sha256": base.digest(amendment), "original_plan_sha256": base.digest(plan),
                      "original_rounds": parent["rounds"], "selection": selection, "fresh_display": display,
                      "new_proposal_calls": 0,
                      "limitations": ["Post-observation handling of truncation; not the original completed plan.",
                          "Two familiar development tasks and imperfect point proxies; no effectiveness inference.",
                          "Missing display cells receive no aggregate; no winner replacement from display outcomes."]}
            base.save_new(work / "report.json", report)
            rendered = _trace(report)
            trace = work / "trace.md"
            if trace.exists():
                base.require(trace.read_text() == rendered, "Continuation trace differs")
            else:
                with trace.open("x") as stream:
                    stream.write(rendered)
            base.save_state(work / "state.json", {"status": status, "workflow_complete": True,
                            "fresh_display_complete": display_complete, "original_plan_complete": False})
            return report
        except BaseException as error:
            base.save_state(work / "state.json", {"status": "incomplete", "workflow_complete": False,
                            "original_plan_complete": False, "error_type": type(error).__name__})
            raise


def replay_continuation(source, output):
    """Reproduce a completed amended workflow from saved exchanges; no transport."""
    source = Path(source).resolve()
    manifest = base.read(source / "continuation/manifest.json")
    original_report = base.read(source / "continuation/report.json")
    base.require(manifest["workflow_sha256"] == base.file_sha(__file__)
                 and manifest["original_engine_sha256"] == base.file_sha(BASE_PATH)
                 and original_report["workflow_complete"] is True
                 and original_report["execution"] == manifest["execution"], "Replay source is incomplete or changed")
    base.require(base.read(source / "continuation/selection.json") == original_report["selection"]
                 and (source / "continuation/trace.md").read_text() == _trace(original_report),
                 "Replay source selection or presentation trace differs")
    hashes = {str(p.relative_to(source)): base.file_sha(p) for p in sorted(source.rglob("*.json"))}
    def saved(request):
        prefix = source / "trace/calls" / request["request_id"]
        base.require(base.read(prefix.with_suffix(".request.json")) == request, "Replay request differs")
        truncated = prefix.with_suffix(".truncation.json")
        if truncated.exists():
            receipt = validate_truncation(request, base.read(truncated))
            base.require(base.read(prefix.with_suffix(".truncation-sha256.json")) == {
                "request_id": request["request_id"], "truncation_sha256": base.file_sha(truncated)}, "Replay truncation changed")
            raise CellTruncated(request["request_id"], receipt["api_request_id"], receipt)
        return _bound_response(source / "trace/calls", request)
    result = run_continuation(manifest["original_plan"], manifest["parent_trace"], output, saved,
                              amendment=manifest["amendment"], evidence_mode="replay", adapter_id="saved-exchanges-v1",
                              source_provenance={"path": str(source), "files_sha256": hashes})
    base.require(set(result) == set(original_report), "Replay report fields differ")
    for key in set(result) - {"execution"}:
        base.require(result[key] == original_report[key], "Replay does not reproduce original " + key)
    return result


def verify_replay(source):
    """Verify the final scientific trace read-only, with temporary offline replay."""
    source = Path(source).resolve()
    with tempfile.TemporaryDirectory(prefix="prompt-continuation-replay-") as directory:
        replay_continuation(source, Path(directory) / "replay")
    return {"schema": SCHEMA + "-replay-verification", "verified": True, "new_service_calls": 0,
            "report_sha256": base.file_sha(source / "continuation/report.json"),
            "selection_sha256": base.file_sha(source / "continuation/selection.json"),
            "trace_sha256": base.file_sha(source / "continuation/trace.md")}
