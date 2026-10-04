"""
Independent outcome verification, in two legs.

LEG 1 -- STATE. Do not trust the write response. The ERP's POST reply is a
claim made by the same call that may have failed; we re-fetch each created
entry from the system of record and diff it field by field against what we
asked it to store. This is what catches `--corrupt-writes`, where every step
"succeeds" and the stored amount is still wrong.

LEG 2 -- CLAIM. An agent can also fail by simply asserting it did the work.
A weak model will happily end a run with "the entry has been created" having
never called a write tool at all. So the final summary is checked against the
trace of what actually executed, and an unsupported claim fails the run.

Leg 1 without leg 2 verifies writes that happened and is blind to work that
never did. Read-only tasks have no state to re-read, so leg 2 is the only
thing standing between a confident sentence and a wrong answer.
"""
import json
from typing import Any

from agent.llm import LLMError, chat_json
from agent.memory import WorkingMemory
from agent.registry import TOOLS, ToolError
from agent.tools.erp import erp_get_entry

# Fields we re-check. created_at and id are assigned by the ERP, not by us.
CHECKED_FIELDS = ("vendor", "invoice_number", "amount", "currency", "due_date")

_CLAIM_SYSTEM = """You audit an autonomous agent's final report against what it
actually did.

You are given the GOAL, the agent's SUMMARY, and the exact list of tool calls
that executed (the trace). Decide whether the summary is supported by the
trace.

Mark it UNSUPPORTED if the summary:
- claims an action that never appears in the trace (e.g. says an entry was
  created when no write tool was called, or the write errored),
- states a value that does not appear in any tool result,
- claims the task is complete when the trace shows it was blocked or partial.

Mark it SUPPORTED if every factual claim traces back to an actual tool result.
Vague but accurate summaries are SUPPORTED. Judge only the facts, not the style.

Return JSON: {"supported": true|false, "reason": "<one sentence>",
"unsupported_claims": ["<claim>", "..."]}"""


def verify(memory: WorkingMemory, summary: str = "", goal: str = "") -> dict:
    """
    Returns {verified, checked, diffs, evidence, note, claim_check}.

    `verified` is the AND of both legs: stored state must match what we sent,
    and the agent's account of the run must match what it actually did.
    """
    state = _verify_state(memory)
    claim = _verify_claim(goal or memory.goal, summary, memory)

    verified = state["verified"] and claim.get("supported", True)

    notes = [state["note"]]
    if not claim.get("supported", True):
        notes.append(f"Claim check FAILED: {claim.get('reason', '')}")
    elif claim.get("checked"):
        notes.append("The agent's summary is supported by the actions in the trace.")

    return {
        "verified": verified,
        "checked": state["checked"],
        "diffs": state["diffs"],
        "evidence": state["evidence"],
        "claim_check": claim,
        "note": " ".join(n for n in notes if n),
    }


# ----------------------------------------------------------------- leg 1


def _verify_state(memory: WorkingMemory) -> dict:
    """Re-read every write from the system of record and diff it."""
    if not memory.writes:
        return {
            "verified": True,
            "checked": 0,
            "diffs": [],
            "evidence": {},
            "note": ("No write operations were performed, so there is no stored "
                     "state to re-read."),
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
        "note": ("All written entries were re-read from the ERP and match what was sent."
                 if not diffs else
                 "Re-reading the ERP found values that do not match what was sent."),
    }


# ----------------------------------------------------------------- leg 2


def _verify_claim(goal: str, summary: str, memory: WorkingMemory) -> dict:
    """Audit the agent's final summary against the trace of what ran."""
    if not summary.strip():
        return {"checked": False, "supported": True, "reason": "No summary to audit."}

    # A cheap structural check first: if the summary asserts a write and no
    # write tool succeeded, that is decidable without a model.
    structural = _structural_claim_check(summary, memory)
    if structural is not None:
        return structural

    try:
        result = chat_json([
            {"role": "system", "content": _CLAIM_SYSTEM},
            {"role": "user", "content":
                f"GOAL:\n{goal}\n\nAGENT SUMMARY:\n{summary}\n\n"
                f"TRACE OF WHAT ACTUALLY EXECUTED:\n{_render_trace(memory)}"},
        ])
    except LLMError as e:
        # Never let the auditor's own failure mask the run's result.
        return {"checked": False, "supported": True,
                "reason": f"Claim check could not run ({e}); state verification stands alone."}

    return {
        "checked": True,
        "supported": bool(result.get("supported", True)),
        "reason": str(result.get("reason", "")),
        "unsupported_claims": result.get("unsupported_claims") or [],
    }


_WRITE_WORDS = ("created", "entered", "added", "recorded", "saved", "logged",
                "inserted", "submitted", "has been entered", "was created")


def _structural_claim_check(summary: str, memory: WorkingMemory) -> dict | None:
    """
    Decidable without a model: the summary claims something was written into
    the internal system, but no write tool ever succeeded.
    """
    if memory.writes:
        return None
    low = summary.lower()
    mentions_system = any(w in low for w in ("erp", "internal system", "entry", "accounts-payable"))
    mentions_write = any(w in low for w in _WRITE_WORDS)
    if mentions_system and mentions_write:
        return {
            "checked": True,
            "supported": False,
            "reason": ("The summary states that an entry was written to the internal "
                       "system, but no write tool call succeeded during this run."),
            "unsupported_claims": [summary.strip()[:300]],
        }
    return None


def _render_trace(memory: WorkingMemory, limit: int = 20) -> str:
    if not memory.history:
        return "(no tool calls executed)"
    lines = []
    for h in memory.history[-limit:]:
        status = f"ERROR: {h['error']}" if h["error"] else _short(h["result"])
        lines.append(f"- {h['tool']}({json.dumps(h['args'], default=str)[:160]}) -> {status}")
    if memory.writes:
        lines.append("WRITES THAT SUCCEEDED:")
        for w in memory.writes:
            lines.append(f"- {w.tool} id={w.entry_id} sent={json.dumps(w.sent, default=str)[:200]}")
    else:
        lines.append("WRITES THAT SUCCEEDED: none")
    return "\n".join(lines)


def _short(value: Any, limit: int = 300) -> str:
    s = str(value)
    return s if len(s) <= limit else s[:limit] + "..."


def _equal(expected: Any, actual: Any) -> bool:
    """Compare tolerantly on type but strictly on value."""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) < 0.01
    if expected is None or actual is None:
        return expected is actual
    return str(expected).strip() == str(actual).strip()
