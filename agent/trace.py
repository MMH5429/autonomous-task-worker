"""
Append-only JSONL trace. This file IS the evidence the run returns.

Every phase and every tool call writes exactly one record. Nothing is
summarised away and failures are never swallowed -- a 500 from the ERP appears
in the trace alongside the retry that recovered from it, because "what went
wrong and how it recovered" is the interesting part of an autonomous run.
"""
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional


class Trace:
    def __init__(self, path: Optional[str] = None):
        if path is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join("runs", f"{stamp}.jsonl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._step = 0

    def record(
        self,
        phase: str,
        *,
        tool: Optional[str] = None,
        tool_input: Any = None,
        output: Any = None,
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
        **extra,
    ) -> None:
        self._step += 1
        rec = {
            "step": self._step,
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "tool": tool,
            "input": _safe(tool_input),
            "output": _safe(output),
            "error": error,
            "duration_ms": duration_ms,
        }
        rec.update({k: _safe(v) for k, v in extra.items()})
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @property
    def steps(self) -> int:
        return self._step


class timed:
    """Context manager returning elapsed milliseconds, for trace records."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = int((time.perf_counter() - self._t0) * 1000)
        return False


def _safe(value: Any) -> Any:
    """Make a value JSON-serialisable, truncating anything enormous."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 4000 else value[:4000] + f"...<truncated {len(value)} chars>"
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return str(value)
