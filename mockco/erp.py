"""
Simulated company ERP.

This is the "internal system" the agent writes into. It is a real FastAPI app
backed by a real SQLite file -- the agent makes real HTTP calls against it and
real rows land on disk. The *company* is simulated; the agent's I/O is not.

Two fault-injection flags exist purely to exercise the agent's reliability and
verification paths:

  --flaky           the FIRST POST /entries of the process returns 500 and does
                    not persist. Subsequent POSTs behave normally.
  --corrupt-writes  stores amount as 0.0 while returning the CORRECT amount in
                    the POST response. A client that trusts its own write
                    response will never notice. This exists to prove that
                    verification re-reads from the system of record.
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

DB_PATH = os.path.join(os.path.dirname(__file__), "erp.db")

# Fault injection, set from argv at startup (see __main__ below) or env.
FLAKY = os.environ.get("MOCKCO_FLAKY") == "1"
CORRUPT_WRITES = os.environ.get("MOCKCO_CORRUPT_WRITES") == "1"

# Module-level counter backing --flaky. Resets when the process restarts, which
# is what makes the flaky demo reproducible.
_post_count = 0

app = FastAPI(title="MockCo ERP", version="1.0.0")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                vendor         TEXT NOT NULL,
                invoice_number TEXT NOT NULL,
                amount         REAL NOT NULL,
                currency       TEXT NOT NULL,
                due_date       TEXT NOT NULL,
                created_at     TEXT NOT NULL
            )
            """
        )


class EntryIn(BaseModel):
    vendor: str = Field(min_length=1)
    invoice_number: str = Field(min_length=1)
    amount: float
    currency: str = Field(min_length=1, max_length=8)
    due_date: str = Field(min_length=1)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "flaky": FLAKY,
        "corrupt_writes": CORRUPT_WRITES,
        "posts_seen": _post_count,
    }


@app.post("/entries")
def create_entry(entry: EntryIn):
    global _post_count
    _post_count += 1

    # --flaky: fail the first write of the process, persisting nothing.
    if FLAKY and _post_count == 1:
        return JSONResponse(status_code=500, content={"error": "upstream timeout"})

    # --corrupt-writes: silently store the wrong amount.
    stored_amount = 0.0 if CORRUPT_WRITES else entry.amount
    created_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO entries
                 (vendor, invoice_number, amount, currency, due_date, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                entry.vendor,
                entry.invoice_number,
                stored_amount,
                entry.currency,
                entry.due_date,
                created_at,
            ),
        )
        entry_id = cur.lastrowid

    # Note: we echo back the amount the CLIENT sent, not what we stored. Under
    # --corrupt-writes these differ, and only an independent re-read reveals it.
    return {
        "id": entry_id,
        "vendor": entry.vendor,
        "invoice_number": entry.invoice_number,
        "amount": entry.amount,
        "currency": entry.currency,
        "due_date": entry.due_date,
        "created_at": created_at,
    }


@app.get("/entries/{entry_id}")
def get_entry(entry_id: int):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"entry {entry_id} not found")
    return dict(row)


@app.get("/entries")
def list_entries(vendor: Optional[str] = None):
    sql = "SELECT * FROM entries"
    args: tuple = ()
    if vendor:
        sql += " WHERE vendor = ?"
        args = (vendor,)
    sql += " ORDER BY id"
    with _connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return {"entries": [dict(r) for r in rows]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the simulated company ERP.")
    parser.add_argument("--flaky", action="store_true",
                        help="first POST /entries returns 500 and does not persist")
    parser.add_argument("--corrupt-writes", action="store_true",
                        help="store amount as 0.0 while returning the correct value")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--reset", action="store_true", help="delete the db first")
    args = parser.parse_args()

    FLAKY = args.flaky
    CORRUPT_WRITES = args.corrupt_writes

    if args.reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    init_db()

    banner = []
    if FLAKY:
        banner.append("FLAKY (first POST will 500)")
    if CORRUPT_WRITES:
        banner.append("CORRUPT-WRITES (amount stored as 0.0)")
    print(f"MockCo ERP on http://127.0.0.1:{args.port}  "
          f"[{', '.join(banner) if banner else 'normal mode'}]", file=sys.stderr)

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
