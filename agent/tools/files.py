"""File tools over the simulated company's document store."""
import os
from datetime import datetime, timezone

from agent.registry import ToolError, register

INVOICE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mockco", "data", "invoices")
)


@register(
    "list_invoices",
    "List every invoice document available in the company document store. "
    "Returns filename, size and last-modified time. Filenames are not reliable "
    "indicators of content -- read a document to learn its vendor and dates.",
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
        })
    return out


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
