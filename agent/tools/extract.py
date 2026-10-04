"""
Schema-constrained extraction from invoice text.

The hard requirement here is NOT accuracy, it is honesty: a field genuinely
absent from the document must come back null. An ERP entry with a hallucinated
due date is worse than no entry at all, because nobody downstream can tell it
is wrong. The model is instructed to emit null, and then Python enforces the
shape regardless of what the model did.
"""
from agent.llm import chat_json
from agent.registry import register
from agent.tools.files import read_invoice

FIELDS = ["vendor", "invoice_number", "amount", "currency", "issue_date", "due_date"]

_PROMPT = """You extract structured fields from invoice documents.

Return a JSON object with exactly these keys:
  vendor          - the issuing company's name, as written
  invoice_number  - the invoice identifier
  amount          - the TOTAL payable, as a number (no currency symbol, no commas)
  currency        - ISO code such as USD or INR
  issue_date      - ISO date YYYY-MM-DD
  due_date        - ISO date YYYY-MM-DD

CRITICAL RULE: if a field is not stated in the document, return null for it.
Do not infer it, do not compute it from payment terms, do not guess a plausible
value. A null is a correct answer. An invented value is a serious error.

Return only the JSON object."""


@register(
    "extract_invoice_fields",
    "Extract vendor, invoice_number, amount, currency, issue_date and due_date "
    "from an invoice, given its FILENAME. The tool reads the document itself, so "
    "you do not need to read it first or pass its text. Any field not present in "
    "the document is returned as null -- it is never guessed. Always check the "
    "returned missing_fields before using the result.",
    {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename from list_invoices, e.g. NWL-2026-0915.txt",
            }
        },
        "required": ["filename"],
    },
)
def extract_invoice_fields(filename: str) -> dict:
    # The tool reads the document itself rather than accepting pasted text.
    # Passing a whole invoice back through the model as a tool argument wastes
    # a large amount of context and invites transcription errors -- the model
    # should pass a reference, not a copy.
    text = read_invoice(filename)
    raw = chat_json([
        {"role": "system", "content": _PROMPT},
        {"role": "user", "content": text},
    ])
    out = _coerce(raw)
    out["source_file"] = filename
    return out


def _coerce(raw: dict) -> dict:
    """
    Normalise the model's object to exactly our schema.

    Missing keys become null rather than being dropped, and strings that only
    *look* like a value ("N/A", "unknown", "") become null too -- models reach
    for those instead of emitting null, and downstream they would read as real
    data. Pure function, so it is directly unit-testable.
    """
    NULLISH = {"", "n/a", "na", "none", "null", "unknown", "not specified",
               "not stated", "tbd", "-"}
    out: dict = {}
    for key in FIELDS:
        val = raw.get(key)
        if isinstance(val, str) and val.strip().lower() in NULLISH:
            val = None
        if key == "amount" and isinstance(val, str):
            cleaned = val.replace(",", "").replace("$", "").strip()
            try:
                val = float(cleaned)
            except ValueError:
                val = None
        out[key] = val
    out["missing_fields"] = [k for k in FIELDS if out[k] is None]
    return out
