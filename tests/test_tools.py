"""
Offline, deterministic tests for the two properties the whole design rests on:

  1. A tool cannot be talked into reading outside its sandbox.
  2. Verification actually catches a mismatch between what was sent and what
     the system of record stored.

Neither test needs the network or an LLM, so they run in under a second and
cannot flake.
"""
import pytest

from agent.memory import WorkingMemory
from agent.registry import ToolError
from agent.tools.extract import _coerce
from agent.tools.files import read_invoice


class TestPathTraversal:
    def test_reads_a_legitimate_invoice(self):
        text = read_invoice("NWL-2026-0117.txt")
        assert "NORTHWIND LOGISTICS" in text

    @pytest.mark.parametrize("attempt", [
        "../../.env",
        "../../../etc/passwd",
        "..\\..\\.env",
        "subdir/../../../secrets.txt",
    ])
    def test_rejects_paths_outside_the_invoice_directory(self, attempt):
        with pytest.raises(ToolError) as exc:
            read_invoice(attempt)
        assert "outside" in str(exc.value).lower() or "no such invoice" in str(exc.value).lower()


class TestExtractionNeverGuesses:
    def test_absent_field_becomes_null_not_a_guess(self):
        # The model omitted due_date entirely because the document lacks one.
        out = _coerce({
            "vendor": "Vertex Paper Co.",
            "invoice_number": "VPX-2026-0727",
            "amount": 21305.00,
            "currency": "USD",
            "issue_date": "2026-07-27",
        })
        assert out["due_date"] is None
        assert "due_date" in out["missing_fields"]

    def test_placeholder_strings_are_treated_as_missing(self):
        # Models reach for "N/A" instead of null; downstream that reads as data.
        out = _coerce({"vendor": "Acme", "invoice_number": "A-1", "amount": "1,200.50",
                       "currency": "USD", "issue_date": "2026-03-02", "due_date": "N/A"})
        assert out["due_date"] is None
        assert out["amount"] == 1200.50  # commas stripped, parsed as a number


class TestVerificationCatchesMismatch:
    """The --corrupt-writes scenario, without needing the server."""

    def _memory_with_write(self, sent, stored, monkeypatch):
        mem = WorkingMemory(goal="test")
        mem.log_write("erp_create_entry", sent=sent, returned={"id": 7, **sent})
        import agent.verify as verify_mod
        monkeypatch.setattr(verify_mod, "erp_get_entry", lambda entry_id: stored)
        return mem

    def test_matching_row_verifies(self, monkeypatch):
        import agent.verify as verify_mod
        sent = {"vendor": "Northwind Logistics", "invoice_number": "NWL-2026-0915",
                "amount": 63120.75, "currency": "USD", "due_date": "2026-10-15"}
        mem = self._memory_with_write(sent, {"id": 7, **sent}, monkeypatch)
        report = verify_mod.verify(mem)
        assert report["verified"] is True
        assert report["diffs"] == []

    def test_corrupted_amount_is_caught(self, monkeypatch):
        import agent.verify as verify_mod
        sent = {"vendor": "Northwind Logistics", "invoice_number": "NWL-2026-0915",
                "amount": 63120.75, "currency": "USD", "due_date": "2026-10-15"}
        stored = {"id": 7, **sent, "amount": 0.0}  # what --corrupt-writes does
        mem = self._memory_with_write(sent, stored, monkeypatch)
        report = verify_mod.verify(mem)

        assert report["verified"] is False
        assert len(report["diffs"]) == 1
        diff = report["diffs"][0]
        assert diff["field"] == "amount"
        assert diff["expected"] == 63120.75
        assert diff["stored"] == 0.0

    def test_read_only_run_is_not_a_failure(self):
        import agent.verify as verify_mod
        report = verify_mod.verify(WorkingMemory(goal="summarise overdue invoices"))
        assert report["verified"] is True
        assert report["checked"] == 0
        assert "read-only" in report["note"].lower()
