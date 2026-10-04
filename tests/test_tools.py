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
        assert "no write operations" in report["note"].lower()


class TestClaimCheck:
    """An agent that asserts it did work it never did must not pass."""

    def test_claiming_a_write_that_never_happened_is_unsupported(self):
        from agent.verify import _structural_claim_check
        mem = WorkingMemory(goal="enter the invoice into our internal system")
        mem.log_step("read_invoice", {"filename": "NWL-2026-0915.txt"}, "...text...")
        result = _structural_claim_check(
            "The accounts-payable entry has been successfully created in the ERP system.",
            mem,
        )
        assert result is not None
        assert result["supported"] is False

    def test_a_real_write_is_not_flagged(self):
        from agent.verify import _structural_claim_check
        mem = WorkingMemory(goal="enter the invoice")
        mem.log_write("erp_create_entry", sent={"amount": 1.0}, returned={"id": 3})
        assert _structural_claim_check("The entry was created in the ERP.", mem) is None

    def test_read_only_summary_is_not_flagged(self):
        from agent.verify import _structural_claim_check
        mem = WorkingMemory(goal="which invoices are overdue")
        assert _structural_claim_check(
            "Three invoices are overdue: NWL-2026-0117, ACM-2026-0342 and OCS-2026-0811.",
            mem,
        ) is None


class TestArgumentBinding:
    """Models invent extra fields and drop required ones. Neither should crash."""

    def _tool(self):
        import agent.loop  # noqa: F401  (registers tools)
        from agent.registry import TOOLS
        return TOOLS["erp_create_entry"]

    def test_unknown_arguments_are_dropped(self):
        from agent.registry import bind_args
        bound = bind_args(self._tool(), {
            "vendor": "Acme", "invoice_number": "A-1", "amount": 1.0,
            "currency": "USD", "due_date": "2026-01-01",
            "confidence": 0.9, "notes": "model invented this",
        })
        assert set(bound) == {"vendor", "invoice_number", "amount", "currency", "due_date"}

    def test_missing_required_argument_is_a_correctable_error(self):
        from agent.registry import bind_args
        with pytest.raises(ToolError) as exc:
            bind_args(self._tool(), {"vendor": "Acme"})
        msg = str(exc.value)
        assert "missing required argument" in msg
        assert "due_date" in msg  # the message tells the model what to supply
