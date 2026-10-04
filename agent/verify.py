"""
Independent outcome verification.

The central rule: do not trust the write response. The ERP's POST reply is a
claim made by the same call that may have failed; verification re-fetches each
created entry from the system of record and diffs it field by field against
what we asked it to store.

This is what catches `--corrupt-writes`, where every step "succeeds" and the
stored amount is still wrong. If the diff is non-empty the run outcome is
FAILED, regardless of how cleanly the steps ran.
"""
from typing import Any

from agent.memory import WorkingMemory
from agent.registry import ToolError
from agent.tools.erp import erp_get_entry

# Fields we re-check. created_at and id are assigned by the ERP, not by us.
CHECKED_FIELDS = ("vendor", "invoice_number", "amount", "currency", "due_date")


def verify(memory: WorkingMemory) -> dict:
    """
    Returns {verified, checked, diffs, evidence, note}.

    A run with no writes is not a failure -- read-only tasks have nothing to
    verify against a system of record, and we say so explicitly rather than
    silently reporting success.
    """
    if not memory.writes:
        return {
            "verified": True,
            "checked": 0,
            "diffs": [],
            "evidence": {},
            "note": ("No write operations were performed, so there is no stored "
                     "state to verify. This was a read-only task; the answer "
                     "rests on the facts recorded in the trace."),
        }

    diffs: list[dict] = []
    evidence: dict[str, Any] = {}

    for w in memory.writes:
        if w.entry_id is None:
            diffs.append({
                "entry_id": None,
                "field": "<id>",
                "expected": "an id returned by the ERP",
                "stored": None,
                "detail": "The write returned no id, so it cannot be confirmed to exist.",
            })
            continue

        try:
            stored = erp_get_entry(w.entry_id)
        except ToolError as e:
            diffs.append({
                "entry_id": w.entry_id,
                "field": "<entire row>",
                "expected": w.sent,
                "stored": None,
                "detail": f"Could not re-read the entry: {e}",
            })
            continue

        evidence[f"entry_{w.entry_id}"] = stored

        for field in CHECKED_FIELDS:
            if field not in w.sent:
                continue
            expected, actual = w.sent[field], stored.get(field)
            if not _equal(expected, actual):
                diffs.append({
                    "entry_id": w.entry_id,
                    "field": field,
                    "expected": expected,
                    "stored": actual,
                    "detail": "Value in the system of record does not match what was sent.",
                })

    return {
        "verified": not diffs,
        "checked": len(memory.writes),
        "diffs": diffs,
        "evidence": evidence,
        "note": ("All written entries were re-read from the ERP and match."
                 if not diffs else
                 "Re-reading the ERP found values that do not match what was sent."),
    }


def _equal(expected: Any, actual: Any) -> bool:
    """Compare tolerantly on type but strictly on value."""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) < 0.01
    if expected is None or actual is None:
        return expected is actual
    return str(expected).strip() == str(actual).strip()
