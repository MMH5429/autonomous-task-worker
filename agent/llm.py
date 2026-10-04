"""
The only place in this codebase that knows which LLM provider we are talking to.

Both supported providers speak the OpenAI chat-completions wire format, so the
adapter is a base URL, a model name and an auth header. Swapping provider is an
env var, not a refactor -- which is how a prototype survives losing an API key
an hour before a deadline.
"""
import json
import os
import re
import time
from typing import Any, Optional

import httpx

from dotenv import load_dotenv

load_dotenv()


class LLMError(RuntimeError):
    """Raised when the provider cannot be reached or returns an unusable reply."""


def _provider_config() -> tuple[str, str, dict]:
    """Return (url, model, headers) for the configured provider."""
    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()

    if provider == "groq":
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise LLMError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "and put it in .env, or set LLM_PROVIDER=ollama to run offline."
            )
        return (
            "https://api.groq.com/openai/v1/chat/completions",
            os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            {"Authorization": f"Bearer {key}"},
        )

    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        return (
            f"{host}/v1/chat/completions",
            os.environ.get("OLLAMA_MODEL", "qwen2.5"),
            {},
        )

    raise LLMError(f"Unknown LLM_PROVIDER={provider!r}. Use 'groq' or 'ollama'.")


def chat(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    json_mode: bool = False,
    temperature: float = 0.0,
    max_retries: int = 6,
) -> dict:
    """
    One chat completion. Returns the raw `message` object from the provider
    (so callers can read `.content` and `.tool_calls` themselves).

    Retries on 429 and 5xx with exponential backoff -- rate limits on a free
    tier are expected, not exceptional.
    """
    url, model, headers = _provider_config()

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    delay = 1.0
    last_err = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=90.0) as client:
                resp = client.post(url, headers=headers, json=payload)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if attempt < max_retries - 1:
                    # Free tiers are token-per-minute limited and the server
                    # tells us exactly how long to wait. Honour that instead of
                    # guessing with a blind backoff.
                    wait = _retry_after(resp) or delay
                    print(f"      [llm] rate limited, waiting {wait:.1f}s "
                          f"(attempt {attempt + 2}/{max_retries})")
                    time.sleep(wait)
                    delay *= 2
                    continue
                raise LLMError(f"LLM provider failed after {max_retries} attempts. {last_err}")
            if resp.status_code == 400:
                salvaged = _salvage_failed_tool_call(resp)
                if salvaged is not None:
                    return salvaged
                raise LLMError(f"LLM provider returned HTTP 400: {resp.text[:500]}")
            if resp.status_code != 200:
                raise LLMError(f"LLM provider returned HTTP {resp.status_code}: {resp.text[:500]}")
            return resp.json()["choices"][0]["message"]
        except httpx.RequestError as e:
            last_err = str(e)
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise LLMError(f"Could not reach LLM provider at {url}: {e}") from e

    raise LLMError(f"LLM call failed: {last_err}")


def _salvage_failed_tool_call(resp) -> dict | None:
    """
    Recover from a provider-side tool-call rejection.

    Some models (the gpt-oss family in particular) emit an out-of-band
    `commentary` channel as if it were a tool. The provider rejects the whole
    completion with 400 tool_use_failed, but it hands back the generated text
    in `failed_generation` -- which is usually the finish payload. Discarding a
    completed run over a provider quirk would be the wrong trade, so we return
    the salvaged text as an ordinary assistant message and let the caller (and
    the claim check) decide what it is worth.
    """
    try:
        err = resp.json().get("error", {})
    except Exception:
        return None
    if err.get("code") != "tool_use_failed":
        return None
    generation = err.get("failed_generation")
    if not generation:
        return None
    return {"role": "assistant", "content": generation, "tool_calls": None}


def _retry_after(resp) -> float | None:
    """Seconds to wait, from the Retry-After header or the error message body."""
    header = resp.headers.get("retry-after")
    if header:
        try:
            return min(float(header) + 0.5, 30.0)
        except ValueError:
            pass
    m = re.search(r"try again in ([0-9.]+)\s*s", resp.text or "", re.I)
    if m:
        return min(float(m.group(1)) + 0.5, 30.0)
    m = re.search(r"try again in ([0-9.]+)m([0-9.]+)s", resp.text or "", re.I)
    if m:
        return min(float(m.group(1)) * 60 + float(m.group(2)) + 0.5, 30.0)
    return None


def chat_json(messages: list[dict], **kwargs) -> dict:
    """A chat completion that must return a JSON object. Parses it or raises."""
    msg = chat(messages, json_mode=True, **kwargs)
    content = (msg.get("content") or "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Some models wrap JSON in prose or a fence; salvage the outermost object.
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise LLMError(f"Expected JSON from the model, got: {content[:300]}") from e


def describe_provider() -> str:
    """Human-readable provider summary, printed at the top of every run."""
    try:
        _, model, _ = _provider_config()
        return f"{os.environ.get('LLM_PROVIDER', 'groq')}:{model}"
    except LLMError:
        return "unconfigured"
