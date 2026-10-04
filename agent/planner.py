"""
Planning. The plan is produced by the model from the tool registry -- there is
no step list anywhere in this codebase.

That is the point. A hardcoded DAG would run the invoice task perfectly and
teach us nothing about autonomy; the test is whether a goal nobody anticipated
still produces a sensible sequence of actions from the same binary.
"""
from agent.llm import chat_json
from agent.memory import WorkingMemory
from agent.registry import catalogue

_PLAN_SYSTEM = """You are the planning component of an autonomous AI worker
operating inside a company. You are given a goal in natural language and the
tools available to you. You produce a short ordered plan.

TOOLS AVAILABLE:
{tools}

Rules:
- Plan only with the tools listed. Do not invent tools or capabilities.
- Prefer the fewest steps that genuinely achieve the goal.
- The goal may be ambiguous (e.g. "the latest invoice" when several exist).
  Resolve ambiguity by INSPECTING data, not by assuming -- e.g. read the
  candidate documents and compare their dates.
- Information that is missing from the documents must be obtained via
  ask_human, never guessed.
- Any step that writes to a company system will additionally require human
  approval at execution time. Account for that.
- Finish with a step that verifies the outcome where a write occurred.

Return JSON of exactly this shape:
{{"objective": "<one sentence restating what success means>",
  "success_criteria": ["<observable condition>", "..."],
  "steps": [{{"n": 1, "intent": "<what this step achieves>", "tool": "<expected tool name or null>"}}]}}"""

_REVISE_SYSTEM = """You are the planning component of an autonomous AI worker.
An earlier plan has run into a problem. Produce a REVISED plan for the
remaining work.

TOOLS AVAILABLE:
{tools}

Rules:
- Do not repeat work that already succeeded.
- Address the specific failure described. If an approach cannot work, choose a
  different one rather than retrying it unchanged.
- If the blocker is missing information that only a human has, plan to use
  ask_human.
- If the goal genuinely cannot be achieved with these tools, say so in the
  objective and return an empty steps list.

Return JSON of exactly this shape:
{{"objective": "<restated objective for the remaining work>",
  "success_criteria": ["..."],
  "steps": [{{"n": 1, "intent": "...", "tool": "<tool name or null>"}}]}}"""


def make_plan(goal: str, memory: WorkingMemory) -> dict:
    return _normalise(chat_json([
        {"role": "system", "content": _PLAN_SYSTEM.format(tools=catalogue())},
        {"role": "user", "content": f"GOAL: {goal}"},
    ]))


def revise_plan(goal: str, plan: dict, failure: str, memory: WorkingMemory) -> dict:
    user = (
        f"ORIGINAL GOAL: {goal}\n\n"
        f"THE PLAN THAT WAS RUNNING:\n{_render(plan)}\n\n"
        f"WHAT WENT WRONG:\n{failure}\n\n"
        f"CURRENT STATE:\n{memory.summary()}"
    )
    return _normalise(chat_json([
        {"role": "system", "content": _REVISE_SYSTEM.format(tools=catalogue())},
        {"role": "user", "content": user},
    ]))


def _normalise(raw: dict) -> dict:
    """Defend against a model that returns a near-miss shape."""
    steps = raw.get("steps") or []
    clean = []
    for i, s in enumerate(steps, start=1):
        if isinstance(s, str):
            s = {"intent": s, "tool": None}
        clean.append({
            "n": s.get("n", i),
            "intent": str(s.get("intent", "")).strip(),
            "tool": s.get("tool") or None,
        })
    return {
        "objective": str(raw.get("objective", "")).strip(),
        "success_criteria": [str(c) for c in (raw.get("success_criteria") or [])],
        "steps": clean,
    }


def _render(plan: dict) -> str:
    lines = [f"Objective: {plan.get('objective', '')}"]
    for c in plan.get("success_criteria", []):
        lines.append(f"  success: {c}")
    for s in plan.get("steps", []):
        tool = f" [{s['tool']}]" if s.get("tool") else ""
        lines.append(f"  {s['n']}. {s['intent']}{tool}")
    return "\n".join(lines)


render = _render
