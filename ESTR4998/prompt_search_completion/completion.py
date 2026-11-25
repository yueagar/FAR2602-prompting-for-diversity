"""Offline attempt accounting for the frozen, fresh-display completion stage."""
from email.utils import parsedate_to_datetime
import re


def additional_attempts(parent_records, completed_records, allowances, *, inherited_retry_id=None):
    """Extract only the attempt suffix covered by the frozen completion targets.

    The caller supplies the two snapshots and the request-ID-to-attempt-cap
    mapping. One specified inherited failure can gain a single attempt. Other
    inherited records, including their request metadata, must match exactly.
    """
    def indexed(records):
        result = {}
        for record in records:
            identity = record["id"]
            if identity in result:
                raise ValueError("Duplicate request identity")
            result[identity] = record
        return result

    parent = indexed(parent_records)
    completed = indexed(completed_records)
    if not parent.keys() <= completed.keys():
        raise ValueError("An inherited request is missing")
    if not all(type(limit) is int and limit > 0 for limit in allowances.values()):
        raise ValueError("Invalid additional attempt allowance")

    tail = []
    for identity, record in completed.items():
        attempts = record["attempts"]
        if identity in parent:
            original = parent[identity]
            prefix = original["attempts"]
            if ({k: v for k, v in record.items() if k != "attempts"}
                    != {k: v for k, v in original.items() if k != "attempts"}
                    or attempts[:len(prefix)] != prefix):
                raise ValueError("Inherited request or attempt prefix changed")
            attempts = attempts[len(prefix):]
            if len(attempts) > int(identity == inherited_retry_id):
                raise ValueError("Inherited retry allowance exceeded")
        elif identity not in allowances:
            raise ValueError("New request outside completion targets")
        if attempts:
            if len(attempts) > allowances.get(identity, 0):
                raise ValueError("Additional attempt allowance exceeded")
            tail.append({**record, "attempts": attempts})
    return tail


def retry_after_seconds(value, now):
    """Parse only the standard delay-seconds or HTTP-date, without logging it."""
    if value is None:
        return None
    value = str(value).strip()
    if re.fullmatch(r"[0-9]+", value):
        normalized = value.lstrip("0") or "0"
        # Preserve arbitrarily long numeric hints as digits, then stop for
        # inspection rather than rounding or replacing them with a shorter cap.
        return {"unbounded_delay_seconds": normalized} if len(normalized) > 12 else int(normalized)
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            return None
        return max(0, parsed.timestamp() - now)
    except (ValueError, TypeError, OverflowError):
        return None
