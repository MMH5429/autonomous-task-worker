"""
Client for the simulated company's ERP.

Real HTTP against a real server. Transient 5xx responses are retried with
exponential backoff; after the budget is exhausted the error is raised, not
swallowed, so the loop can observe it and adapt.
"""
import os
import time
from typing import Optional

import httpx

from agent.registry import ToolError, register

BASE_URL = os.environ.get("ERP_BASE_URL", "http://127.0.0.1:8099").rstrip("/")

RETRY_DELAYS = (0.5, 1.0, 2.0)  # three retries after the initial attempt


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{BASE_URL}{path}"
    last = ""
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.request(method, url, **kwargs)
        except httpx.RequestError as e:
            last = f"connection error: {e}"
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
                continue
            raise ToolError(
                f"ERP unreachable at {url} after {attempt + 1} attempts ({last}). "
                "Is the mockco server running?"
            ) from e

        if resp.status_code >= 500:
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if attempt < len(RETRY_DELAYS):
                # Visible on purpose: the retry is part of the story.
                print(f"      [erp] {last} -- retrying in {RETRY_DELAYS[attempt]}s "
                      f"(attempt {attempt + 2}/{len(RETRY_DELAYS) + 1})")
                time.sleep(RETRY_DELAYS[attempt])
                continue
            raise ToolError(f"ERP failed after {attempt + 1} attempts. Last: {last}")

        if resp.status_code == 404:
            raise ToolError(f"ERP returned 404 for {path}: {resp.text[:200]}")
        if resp.status_code >= 400:
            raise ToolError(f"ERP rejected the request ({resp.status_code}): {resp.text[:300]}")
        return resp.json()

    raise ToolError(f"ERP request failed: {last}")


@register(
    "erp_create_entry",
    "Create an accounts-payable entry in the company's internal ERP system. "
    "This is a WRITE: it requires human approval and all five fields must be "
    "real values taken from the document. Never pass a guessed due_date.",
    {
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "invoice_number": {"type": "string"},
            "amount": {"type": "number"},
            "currency": {"type": "string"},
            "due_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
        },
        "required": ["vendor", "invoice_number", "amount", "currency", "due_date"],
    },
    writes=True,
)
def erp_create_entry(vendor: str, invoice_number: str, amount: float,
                     currency: str, due_date: str) -> dict:
    return _request("POST", "/entries", json={
        "vendor": vendor,
        "invoice_number": invoice_number,
        "amount": amount,
        "currency": currency,
        "due_date": due_date,
    })


@register(
    "erp_get_entry",
    "Fetch one ERP entry by its id, as actually stored in the system of record.",
    {
        "type": "object",
        "properties": {"entry_id": {"type": "integer"}},
        "required": ["entry_id"],
    },
)
def erp_get_entry(entry_id: int) -> dict:
    return _request("GET", f"/entries/{int(entry_id)}")


@register(
    "erp_list_entries",
    "List all entries currently in the ERP, optionally filtered by vendor.",
    {
        "type": "object",
        "properties": {"vendor": {"type": "string"}},
        "required": [],
    },
)
def erp_list_entries(vendor: Optional[str] = None) -> dict:
    params = {"vendor": vendor} if vendor else None
    return _request("GET", "/entries", params=params)
