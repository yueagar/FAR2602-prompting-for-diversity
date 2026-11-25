"""Offline scheduling and accounting for the default-only generator screen."""
from decimal import Decimal
from pathlib import Path
import sys

API = Path(__file__).resolve().parents[1] / "api_study"
sys.path.insert(0, str(API))
from api_runtime import no_reply_retryable


def accounting(records):
    attempts = [a for r in records for a in r["attempts"]]
    return {"attempted_requests": len(attempts),
        "known_cost_usd": sum(a.get("reported_cost_usd") or 0 for a in attempts),
        "unknown_cost_attempts": sum(a.get("reported_cost_usd") is None for a in attempts),
        "committed_usd": sum(a.get("reported_cost_usd") if a.get("reported_cost_usd") is not None else a["reserve_usd"] for a in attempts)}


def failed_cost_usd(records):
    """Known failed-attempt charges; unresolved attempts retain their reserve."""
    return sum((Decimal(str(a["reported_cost_usd"] if a.get("reported_cost_usd") is not None else a["reserve_usd"]))
        for record in records for a in record["attempts"] if a["status"] != "ok"), Decimal("0"))


def remaining_jobs(plan, records, *, extended_generation_id=None):
    """Future eligible requests, excluding permanently missing dependencies."""
    limits = {}
    if "retry_amendment" in plan:
        limits = plan["retry_amendment"]["generation_attempt_limits"]
        if extended_generation_id is None or limits != {extended_generation_id: 3}:
            raise ValueError("Retry allowance differs from the single bound request")

    records = {r["id"]: r for r in records}
    unavailable = set()
    for job in plan["jobs"]:
        if job["metadata"]["role"] != "generation": continue
        record = records.get(job["id"])
        if not record or not record["attempts"]: continue
        last = record["attempts"][-1]
        if last["status"] != "ok" and (len(record["attempts"]) >= limits.get(job["id"], 2)
                or not no_reply_retryable(last)):
            unavailable.add(job["id"])
    pending, missing = [], []
    for job in plan["jobs"]:
        identity = job["id"]; record = records.get(identity)
        attempts = record["attempts"] if record else []
        if attempts and attempts[-1]["status"] == "ok": continue
        if identity in unavailable or any(x in unavailable for x in job.get("response_ids", [])):
            missing.append(identity); continue
        limit = limits.get(identity, 2) if job["metadata"]["role"] == "generation" else 2
        if len(attempts) >= limit:
            missing.append(identity); continue
        pending.append({"id": identity, "maximum_additional_attempts": limit - len(attempts)})
    return pending, missing
