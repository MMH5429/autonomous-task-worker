"""File tools over the simulated company's document store."""
import os
from datetime import datetime, timezone

from agent.registry import ToolError, register

INVOICE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mockco", "data", "invoices")
)


@register(
    "list_invoices",
    "List every invoice document in the company document store. Returns each "
    "filename plus a short preview of its opening lines, which is usually "
    "enough to tell which vendor it belongs to. Use the preview to pick the "
    "right document; read or extract it for exact values.",
    {"type": "object", "properties": {}, "required": []},
)
def list_invoices() -> list[dict]:
    if not os.path.isdir(INVOICE_DIR):
        raise ToolError(f"Invoice directory not found: {INVOICE_DIR}")
    out = []
    for fn in sorted(os.listdir(INVOICE_DIR)):
        if not fn.lower().endswith(".txt"):
            continue
        p = os.path.join(INVOICE_DIR, fn)
        st = os.stat(p)
        out.append({
            "filename": fn,
            "size_bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "preview": _preview(p),
        })
    return out


def _preview(path: str, lines: int = 2, width: int = 90) -> str:
    """
    First couple of non-blank lines of a document.

    A real document store has an index; without one the agent has to read every
    file just to find out who issued it, which wastes context and -- as a run
    showed -- tempts it to ask a human a question it could answer itself.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            picked = []
            for raw in fh:
                raw = raw.strip()
                if raw:
                    picked.append(raw[:width])
                if len(picked) >= lines:
                    break
        return " | ".join(picked)
    except OSError:
        return ""


@register(
    "read_invoice",
    "Read the full raw text of one invoice document by filename.",
    {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Filename from list_invoices, e.g. NWL-2026-0117.txt"}
        },
        "required": ["filename"],
    },
)
def read_invoice(filename: str) -> str:
    # Resolve and confine. An agent that can be talked into reading ../../.env
    # is a liability, so the guard lives in the tool rather than the prompt.
    target = os.path.abspath(os.path.join(INVOICE_DIR, filename))
    if os.path.commonpath([target, INVOICE_DIR]) != INVOICE_DIR:
        raise ToolError(
            f"Refused: {filename!r} resolves outside the invoice directory."
        )
    if not os.path.isfile(target):
        raise ToolError(f"No such invoice: {filename}")
    with open(target, "r", encoding="utf-8") as fh:
        return fh.read()
