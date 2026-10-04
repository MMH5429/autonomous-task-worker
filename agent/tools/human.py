"""
Human-in-the-loop.

Two distinct things, deliberately separated:

  ask_human       the agent is MISSING information it cannot derive. Model-facing
                  tool -- the agent decides when it needs to ask.
  request_approval the agent is about to CHANGE something. Not a model-facing
                  tool: the loop enforces it from the registry's `writes` flag,
                  so approval cannot be skipped by a model that decides it knows
                  better.
"""
import json
import sys

from agent.registry import register

# Set by run.py --auto-approve. Only suppresses the approval gate, never the
# agent's own questions -- those still have to be answered.
AUTO_APPROVE = False


@register(
    "ask_human",
    "Ask the human operator for information you need but cannot obtain -- a "
    "value missing from every document, or a business decision only they can "
    "make. Use this instead of guessing. Do NOT use it to confirm, acknowledge "
    "or announce finished work: report that by calling finish. Asking a "
    "question you could answer by reading a document is also wrong.",
    {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "A specific, answerable question"},
            "context": {"type": "string", "description": "What you already established, and why you must ask"},
        },
        "required": ["question", "context"],
    },
)
def ask_human(question: str, context: str) -> dict:
    print("\n  ┌─ AGENT NEEDS INPUT " + "─" * 50)
    print(f"  │ {question}")
    if context:
        for line in _wrap(context, 72):
            print(f"  │   {line}")
    print("  └" + "─" * 70)
    try:
        answer = input("  your answer > ").strip()
    except EOFError:
        answer = ""
    if not answer:
        return {"answered": False, "answer": None,
                "note": "Operator gave no answer. Do not invent a value; report this as blocked."}
    return {"answered": True, "answer": answer}


def request_approval(action_description: str, payload: dict) -> dict:
    """Approval gate before any mutating tool call. Called by the loop, not the model."""
    if AUTO_APPROVE:
        print(f"  [approval] auto-approved (--auto-approve): {action_description}")
        return {"approved": True, "reason": "auto-approved via --auto-approve flag"}

    print("\n  ┌─ APPROVAL REQUIRED " + "─" * 50)
    print(f"  │ {action_description}")
    print("  │ Payload:")
    for line in json.dumps(payload, indent=2).splitlines():
        print(f"  │   {line}")
    print("  └" + "─" * 70)
    try:
        answer = input("  approve? [y/N] > ").strip().lower()
    except EOFError:
        answer = "n"
    if answer in ("y", "yes"):
        return {"approved": True, "reason": "approved by operator"}
    reason = input("  reason for declining (optional) > ").strip() or "declined by operator"
    return {"approved": False, "reason": reason}


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines
