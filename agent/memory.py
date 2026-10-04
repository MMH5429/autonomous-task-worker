"""
Working memory for a single run.

Deliberately not a vector store. What the agent needs to carry forward is
structured and small: facts it discovered, what it has already done, and what
it wrote (so verification knows what to check). Embedding any of that would
make it fuzzier, not better. Cross-run company memory is the next thing to
build -- see the README.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WriteRecord:
    """A write the agent performed, kept so verification can re-check it."""
    tool: str
    sent: dict          # what we asked the system of record to store
    returned: dict      # what the write response claimed
    entry_id: Any       # identifier to re-fetch by


@dataclass
class WorkingMemory:
    goal: str
    facts: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    writes: list[WriteRecord] = field(default_factory=list)

    def remember(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def log_step(self, tool: str, args: dict, result: Any, error: str | None = None) -> None:
        self.history.append(
            {"tool": tool, "args": args, "result": result, "error": error}
        )

    def log_write(self, tool: str, sent: dict, returned: dict) -> None:
        entry_id = returned.get("id") if isinstance(returned, dict) else None
        self.writes.append(
            WriteRecord(tool=tool, sent=sent, returned=returned, entry_id=entry_id)
        )

    def summary(self, max_steps: int = 12) -> str:
        """Compact state passed to the planner. Keeps prompts small and legible."""
        lines = [f"GOAL: {self.goal}"]
        if self.facts:
            lines.append("FACTS DISCOVERED THIS RUN:")
            for k, v in self.facts.items():
                lines.append(f"  - {k}: {_short(v)}")
        if self.history:
            lines.append("ACTIONS ALREADY TAKEN:")
            for h in self.history[-max_steps:]:
                status = f"ERROR: {h['error']}" if h["error"] else _short(h["result"])
                lines.append(f"  - {h['tool']}({_short(h['args'])}) -> {status}")
        if self.writes:
            lines.append("WRITES PERFORMED (will be independently verified):")
            for w in self.writes:
                lines.append(f"  - {w.tool} id={w.entry_id} sent={_short(w.sent)}")
        return "\n".join(lines)


def _short(value: Any, limit: int = 240) -> str:
    s = str(value)
    return s if len(s) <= limit else s[:limit] + "..."
