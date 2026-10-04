#!/usr/bin/env python
"""
CLI entrypoint for the autonomous task worker.

    python run.py "Find the latest invoice from Northwind Logistics, extract the
                   amount and due date, enter it into our internal system, and
                   tell me once it is done."

The same binary runs every task. Nothing about the invoice workflow is special
cased here -- see README "Generalization".
"""
import argparse
import sys
from datetime import date

from agent import llm
from agent.loop import Agent
from agent.trace import Trace


def main() -> int:
    # Windows consoles default to cp1252, which cannot encode characters the
    # model routinely emits (non-breaking hyphens, box drawing, currency signs).
    # Without this a perfectly good run dies in print().
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(
        description="Autonomous AI task worker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("task", help="the task, in natural language")
    ap.add_argument("--auto-approve", action="store_true",
                    help="skip the interactive approval gate before writes (for recorded demos)")
    ap.add_argument("--trace-file", default=None,
                    help="where to write the JSONL trace (default: runs/<timestamp>.jsonl)")
    ap.add_argument("--today", default=date.today().isoformat(),
                    help="the date the agent should treat as today (default: the real date)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-phase output")
    args = ap.parse_args()

    # Import for its module-level flag; the approval gate reads it directly.
    from agent.tools import human
    human.AUTO_APPROVE = args.auto_approve

    trace = Trace(args.trace_file)

    print("=" * 74)
    print("  AUTONOMOUS TASK WORKER")
    print(f"  provider : {llm.describe_provider()}")
    print(f"  today    : {args.today}")
    print(f"  trace    : {trace.path}")
    print("=" * 74)

    try:
        agent = Agent(goal=args.task, today=args.today, trace=trace, verbose=not args.quiet)
        result = agent.run()
    except llm.LLMError as e:
        print(f"\n  LLM provider error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n  interrupted.", file=sys.stderr)
        return 130

    _report(result)
    return 0 if result["outcome"] == "SUCCESS" else 1


def _report(result: dict) -> None:
    v = result["verification"]
    bar = "=" * 74
    print(f"\n{bar}")
    print(f"  OUTCOME: {result['outcome']}")
    print(bar)

    print("\n  WHAT HAPPENED")
    for line in _wrap(result["summary"] or "(no summary returned)", 68):
        print(f"    {line}")

    print("\n  VERIFICATION")
    for line in _wrap(v["note"], 68):
        print(f"    {line}")
    if v["diffs"]:
        print("\n    Differences between what was sent and what is stored:")
        for d in v["diffs"]:
            print(f"      entry {d['entry_id']} field '{d['field']}'")
            print(f"        sent   : {d['expected']!r}")
            print(f"        stored : {d['stored']!r}")

    if v["evidence"]:
        print("\n  EVIDENCE (read back from the ERP, not from the write response)")
        for key, row in v["evidence"].items():
            print(f"    {key}:")
            for k, val in row.items():
                print(f"      {k:<15}{val}")

    print(f"\n  replans      : {result['replans']}")
    print(f"  duration     : {result['duration_s']}s")
    print(f"  full trace   : {result['trace_file']}")

    if result["outcome"] == "FAILED":
        print("\n  NOTE: every step reported success, but re-reading the system of")
        print("        record shows the stored data is wrong. The run is FAILED.")
    print()


def _wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    for para in str(text).splitlines():
        cur = ""
        for word in para.split():
            if len(cur) + len(word) + 1 > width:
                out.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        out.append(cur)
    return out or [""]


if __name__ == "__main__":
    sys.exit(main())
