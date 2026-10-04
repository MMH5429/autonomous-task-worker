"""
Tool registry.

Capabilities are added by REGISTERING a tool, never by editing the control
loop. That is the property the "generalization" criterion is really asking
about: adding a new company system should touch one file in agent/tools/ and
one register() call, and nothing in loop.py.

`writes=True` marks a tool as mutating. The loop uses that flag -- not the
model's judgement -- to decide when a human approval gate is required.
"""
from dataclasses import dataclass, field
from typing import Callable


class ToolError(RuntimeError):
    """A tool failed in a way the agent is expected to observe and adapt to."""


@dataclass
class Tool:
    name: str
    description: str
    json_schema: dict
    fn: Callable
    writes: bool = False


TOOLS: dict[str, Tool] = {}


def register(name: str, description: str, json_schema: dict, *, writes: bool = False):
    """Decorator: register a function as an agent-callable tool."""
    def wrap(fn: Callable) -> Callable:
        TOOLS[name] = Tool(name=name, description=description,
                           json_schema=json_schema, fn=fn, writes=writes)
        return fn
    return wrap


def to_api_schema() -> list[dict]:
    """Emit the registry in OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.json_schema,
            },
        }
        for t in TOOLS.values()
    ]


def catalogue() -> str:
    """Plain-text tool list for the planner prompt."""
    return "\n".join(
        f"- {t.name}{' [WRITES - requires human approval]' if t.writes else ''}: {t.description}"
        for t in TOOLS.values()
    )


def call(name: str, args: dict):
    if name not in TOOLS:
        raise ToolError(f"No such tool: {name}. Available: {', '.join(TOOLS)}")
    return TOOLS[name].fn(**args)
