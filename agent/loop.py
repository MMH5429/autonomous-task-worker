"""
The control loop:

    Goal -> Understand -> Plan -> Execute -> Observe -> Adapt -> Verify -> Complete

Written as plain Python with explicit control flow and no agent framework.
Every phase is a named function, every phase writes to the trace, and the
approval gate before writes is enforced here from the registry's `writes` flag
rather than being left to the model's discretion.
"""
import json
import time
from typing import Any

from agent import planner
from agent.llm import chat, describe_provider
from agent.memory import WorkingMemory
from agent.registry import TOOLS, ToolError, bind_args, to_api_schema
from agent.tools import erp, extract, files, human  # noqa: F401  (registers tools)
from agent.tools.human import request_approval
from agent.trace import Trace, timed
from agent.verify import verify as verify_outcome

MAX_STEPS = 25

_EXEC_SYSTEM = """You are an autonomous AI worker completing a task inside a company.

You have a goal and a plan. Work through it by calling tools. After each tool
result, decide the next action based on WHAT ACTUALLY HAPPENED, not on what the
plan assumed.

Hard rules:
- Never invent data. If a field is missing from a document, it is missing. Use
  ask_human rather than supplying a plausible value.
- Resolve ambiguity by inspecting data. If the goal says "the latest" and
  several candidates exist, read them and compare their dates.
- Writes require human approval; this is handled for you, but a declined
  approval means you must stop and report, not find another way in.
- When the task is complete, call finish with a concise summary. Do not claim
  success in prose without calling finish.
- If you cannot proceed, call finish with status "blocked" and explain why.

Today's date is {today}."""


def _finish_tool_schema() -> dict:
    """The model's way of declaring the work done. Deliberately not in the
    registry: it ends the loop rather than acting on the world."""
    return {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call when the task is complete, or when you are blocked and cannot proceed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["completed", "blocked"]},
                    "summary": {
                        "type": "string",
                        "description": "Concise account of what you did and what the answer is",
                    },
                },
                "required": ["status", "summary"],
            },
        },
    }


class Agent:
    def __init__(self, goal: str, today: str, trace: Trace, verbose: bool = True):
        self.goal = goal
        self.today = today
        self.trace = trace
        self.verbose = verbose
        self.memory = WorkingMemory(goal=goal)
        self.plan: dict = {}
        self.replans = 0

    # ---------------------------------------------------------------- phases

    def understand(self) -> None:
        """Restate the objective. Success criteria come out of the planning
        call, which needs the tool catalogue to know what success can mean."""
        self._say("UNDERSTAND", f"Goal: {self.goal}")
        self.trace.record("understand", output=self.goal)

    def plan_phase(self) -> dict:
        with timed() as t:
            self.plan = planner.make_plan(self.goal, self.memory)
        self.trace.record("plan", output=self.plan, duration_ms=t.ms)
        self._say("PLAN", planner.render(self.plan))
        self._say("", "(this plan came from the model -- there is no step list in the source)")
        return self.plan

    def adapt(self, failure: str) -> None:
        self.replans += 1
        with timed() as t:
            self.plan = planner.revise_plan(self.goal, self.plan, failure, self.memory)
        self.trace.record("adapt", output=self.plan, duration_ms=t.ms, failure=failure)
        self._say("ADAPT", f"Replanning after: {failure}\n{planner.render(self.plan)}")

    def execute(self) -> dict:
        """
        Run the plan. The model chooses each next tool call given the plan and
        everything observed so far; we execute it, feed the result back, and
        repeat until finish or the step budget runs out.
        """
        messages = [
            {"role": "system", "content": _EXEC_SYSTEM.format(today=self.today)},
            {
                "role": "user",
                "content": f"GOAL: {self.goal}\n\nPLAN:\n{planner.render(self.plan)}\n\nBegin.",
            },
        ]
        tools = to_api_schema() + [_finish_tool_schema()]

        for step in range(1, MAX_STEPS + 1):
            with timed() as t:
                msg = chat(messages, tools=tools)
            self.trace.record("think", output=msg.get("content"), duration_ms=t.ms, step_no=step)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # The model answered in prose instead of finishing. Nudge it,
                # then accept the prose rather than burning the whole budget.
                content = (msg.get("content") or "").strip()
                self._say("OBSERVE", f"(no tool call) {content[:300]}")
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "If the task is complete, call finish. If not, call the next tool.",
                })
                if step > 2 and content:
                    return _as_finish(content)
                continue

            messages.append(msg)

            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "finish":
                    self.trace.record("complete", tool="finish", tool_input=args)
                    return {
                        "status": args.get("status", "completed"),
                        "summary": args.get("summary", ""),
                    }

                result, error = self._run_tool(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(
                        _trim(result if error is None else {"error": error}),
                        ensure_ascii=False,
                        default=str,
                    ),
                })

                if error and self.replans < 3:
                    # Observe -> Adapt. The failure is visible to the model AND
                    # triggers a real replan, bounded so we cannot thrash.
                    self.adapt(f"Tool {name} failed: {error}")
                    messages.append({
                        "role": "user",
                        "content": f"That failed. Revised plan:\n{planner.render(self.plan)}\nContinue.",
                    })

        self.trace.record("complete", output="step budget exhausted")
        return {
            "status": "blocked",
            "summary": f"Step budget of {MAX_STEPS} exhausted before the task completed.",
        }

    # ------------------------------------------------------- execute helpers

    def _run_tool(self, name: str, args: dict) -> tuple[Any, str | None]:
        """Execute one tool call: approval gate, invoke, observe, trace."""
        tool = TOOLS.get(name)

        # Approval gate. Enforced from the registry, not from the prompt, so a
        # model cannot talk its way past it.
        if tool is not None and tool.writes:
            decision = request_approval(f"{name}: write to the company ERP", args)
            self.trace.record("approval", tool=name, tool_input=args, output=decision)
            if not decision["approved"]:
                err = f"Human declined the write: {decision['reason']}"
                self.memory.log_step(name, args, None, error=err)
                self._say("OBSERVE", err)
                return None, err

        self._say("EXECUTE", f"{name}({_compact(args)})")
        if tool is None:
            err = f"No such tool: {name}"
            self.trace.record("execute", tool=name, tool_input=args, error=err)
            self.memory.log_step(name, args, None, error=err)
            self._say("OBSERVE", f"FAILED: {err}")
            return None, err

        try:
            with timed() as t:
                result = tool.fn(**bind_args(tool, args))
        except ToolError as e:
            self.trace.record("execute", tool=name, tool_input=args, error=str(e))
            self.memory.log_step(name, args, None, error=str(e))
            self._say("OBSERVE", f"FAILED: {e}")
            return None, str(e)
        except Exception as e:  # unexpected: still observable, still not fatal
            err = f"{type(e).__name__}: {e}"
            self.trace.record("execute", tool=name, tool_input=args, error=err)
            self.memory.log_step(name, args, None, error=err)
            self._say("OBSERVE", f"FAILED: {err}")
            return None, err

        self.trace.record("execute", tool=name, tool_input=args, output=result, duration_ms=t.ms)
        self.observe(name, args, result)
        return result, None

    def observe(self, name: str, args: dict, result: Any) -> None:
        """Record what the result tells us, and remember what matters later."""
        self.memory.log_step(name, args, result)

        if name == "extract_invoice_fields" and isinstance(result, dict):
            missing = result.get("missing_fields") or []
            key = result.get("invoice_number") or "document"
            self.memory.remember(f"extracted:{key}", result)
            if missing:
                self._say(
                    "OBSERVE",
                    f"extracted {key} -- MISSING from document: {', '.join(missing)} "
                    f"(null, not guessed)",
                )
            else:
                self._say("OBSERVE", f"extracted {key} -- all fields present")
            return

        # A write is remembered so verification can independently re-read it.
        tool = TOOLS.get(name)
        if tool is not None and tool.writes and isinstance(result, dict):
            self.memory.log_write(name, sent=args, returned=result)
            self._say(
                "OBSERVE",
                f"wrote entry id={result.get('id')} (claimed by the ERP; will be re-checked)",
            )
            return

        self._say("OBSERVE", _compact(result))

    # ------------------------------------------------------- verify/complete

    def verify_phase(self, summary: str = "") -> dict:
        with timed() as t:
            report = verify_outcome(self.memory, summary=summary, goal=self.goal)
        self.trace.record("verify", output=report, duration_ms=t.ms)

        if report["checked"] == 0:
            self._say("VERIFY", "no writes to re-read")
        elif report["diffs"]:
            self._say("VERIFY", "MISMATCH between what was sent and what is stored:")
            for d in report["diffs"]:
                self._say(
                    "",
                    f"  entry {d['entry_id']}.{d['field']}: "
                    f"sent {d['expected']!r}, stored {d['stored']!r}",
                )
        else:
            self._say(
                "VERIFY",
                f"re-read {report['checked']} entry/entries from the ERP -- all fields match",
            )

        claim = report.get("claim_check") or {}
        if claim.get("checked"):
            if claim.get("supported"):
                self._say("", "claim check: the summary is supported by the trace")
            else:
                self._say("", f"claim check FAILED: {claim.get('reason', '')}")
        return report

    # ---------------------------------------------------------------- driver

    def run(self) -> dict:
        t0 = time.perf_counter()
        self.trace.record("goal", output=self.goal, provider=describe_provider())
        self.understand()
        self.plan_phase()
        exec_result = self.execute()
        report = self.verify_phase(summary=exec_result.get("summary", ""))

        # The decisive line: a run where every step succeeded is still FAILED if
        # the system of record disagrees with what we believe we wrote.
        if exec_result["status"] == "blocked":
            outcome = "BLOCKED"
        elif not report["verified"]:
            outcome = "FAILED"
        else:
            outcome = "SUCCESS"

        result = {
            "outcome": outcome,
            "summary": exec_result.get("summary", ""),
            "verification": report,
            "replans": self.replans,
            "trace_file": self.trace.path,
            "duration_s": round(time.perf_counter() - t0, 1),
        }
        self.trace.record("outcome", output=result)
        return result

    # ------------------------------------------------------------------ util

    def _say(self, phase: str, text: str) -> None:
        if not self.verbose:
            return
        if phase:
            print(f"\n  [{phase}] {text}")
        else:
            print(f"          {text}")


def _as_finish(content: str) -> dict:
    """
    Accept a finish payload the model wrote as prose instead of calling the
    tool. Models do this often enough that rejecting it would fail runs that
    actually succeeded -- but we parse it rather than trusting the wording,
    and the claim check still audits whatever summary comes out.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split(chr(10), 1)[-1] if chr(10) in text else text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict) and "summary" in obj:
                return {
                    "status": obj.get("status", "completed"),
                    "summary": str(obj["summary"]),
                }
        except json.JSONDecodeError:
            pass
    return {"status": "completed", "summary": content}


def _compact(value: Any, limit: int = 300) -> str:
    s = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit] + "..."


def _trim(value: Any, limit: int = 6000) -> Any:
    """Keep tool results fed back to the model within a sane size."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...<truncated>"
    return value
