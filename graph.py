"""
graph.py
---------
Builds a LangGraph ReAct-style agent wired with Tavily web search.

What this module provides
- `build_agent(model_name: str | None)`:
    Returns a graph-like agent that can be invoked with chat history.
- `invoke_agent(agent, history, preamble_override=None)`:
    Invokes the agent and returns a dict with the final text, tool trace,
    and the raw response. Supports injecting a custom system preamble.

Design choices
- We prefer the positional signature `create_react_agent(model, tools)` to
  avoid keyword-arg breakage across LangGraph minor versions.
- We inject the system preamble as a *system message* at call time, which is
  robust across versions (instead of relying on state/messages modifiers).

Environment variables
- OPENAI_API_KEY
- TAVILY_API_KEY
- (optional) OPENAI_MODEL (default: gpt-4o-mini)
"""

from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Prefer the new Tavily tool package; gracefully fallback if not installed.
try:
    from langchain_tavily import TavilySearch  # recommended package
except Exception:  # pragma: no cover
    from langchain_community.tools.tavily_search import (  # legacy fallback
        TavilySearchResults as TavilySearch
    )

from langgraph.prebuilt import create_react_agent


# Default preamble; callers can override in `invoke_agent`.
PREAMBLE = (
    "You are a helpful research assistant. "
    "When a question likely requires up-to-date information or verification, "
    "use the web search tool first. Synthesize findings and, when you used search, "
    "append a short 'Sources:' section with 1–3 direct URLs. Keep answers concise."
)


def make_preamble(answer_mode: str = "facts") -> str:
    """
    Compose the system preamble with an answer-mode hint.

    Parameters
    ----------
    answer_mode : str
        One of {'facts', 'summary', 'links'}. Controls formatting preference.

    Returns
    -------
    str
        A system message to inject into the conversation.
    """
    mode_hint = {
        "facts": "Write a short, factual answer followed by Sources.",
        "summary": "Write a concise paragraph-style summary followed by Sources.",
        "links": "Do not elaborate; return a one-sentence answer and a bulleted Sources list.",
    }.get(answer_mode, "Write a short, factual answer followed by Sources.")
    return PREAMBLE + " " + mode_hint


def _require_env(var_name: str) -> str:
    """Return the environment variable or raise a helpful error if missing."""
    val = os.getenv(var_name)
    if not val:
        raise RuntimeError(
            f"Missing environment variable: {var_name}. "
            f"Please set it in your .env or your shell session."
        )
    return val


def build_agent(model_name: str | None = None):
    """
    Construct and return a LangGraph ReAct agent with a Tavily search tool.

    Parameters
    ----------
    model_name : str, optional
        OpenAI chat model name to use. If None, falls back to env OPENAI_MODEL
        or 'gpt-4o-mini'.

    Returns
    -------
    agent : Any
        A graph-like object exposing `.invoke({"messages": [...]}) -> dict`.
        Messages should be a list of (role, content) tuples.
    """
    load_dotenv()
    _require_env("OPENAI_API_KEY")
    _require_env("TAVILY_API_KEY")

    resolved_model = model_name or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    model = ChatOpenAI(model=resolved_model, temperature=0)

    search = TavilySearch(max_results=5)

    # Prefer positional signature first; fallback to older kw-arg signature.
    try:
        agent = create_react_agent(model, [search])
    except TypeError:
        agent = create_react_agent(llm=model, tools=[search])  # type: ignore[arg-type]
    return agent


def _normalize_history(
    history: list,
) -> list[tuple[str, str]]:
    """
    Normalize history items to a uniform (role, content) tuple list.

    Supports:
    - [("user", "hi"), ("assistant", "hello")]
    - [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    Anything else will be stringified.
    """
    out: list[tuple[str, str]] = []
    for item in history or []:
        if isinstance(item, tuple) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
        elif isinstance(item, dict):
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
            out.append((role, content))
        else:
            # fallback: treat as user text
            out.append(("user", str(item)))
    return out


def _inject_preamble(raw_history: list, preamble: str) -> list[tuple[str, str]]:
    """
    Ensure the first message is our preamble as a system message.

    We first normalize the history so it can be either list of tuples or list of
    dicts. This makes it safe to call `invoke_agent(...)` with temporary dict-based
    histories (like the chat-summary button).
    """
    history = _normalize_history(raw_history)
    if history and history[0][0] == "system" and preamble in history[0][1]:
        return history
    return [("system", preamble)] + history


def invoke_agent(
    agent,
    history: List[Tuple[str, str]],
    preamble_override: str | None = None,
) -> Dict[str, Any]:
    """
    Run the agent with chat history and return a simplified response object.

    Parameters
    ----------
    agent : Any
        The graph returned from `build_agent()`.
    history : list[tuple[str, str]]
        Chat history in (role, content) format.
    preamble_override : str or None
        Optional system preamble to inject (e.g., with answer-mode hint).

    Returns
    -------
    out : dict
        {
          "final_text": str,        # Assistant's final answer content
          "tool_trace": list[dict], # Best-effort extraction of tool calls
          "raw": dict               # Raw response returned by the agent
        }
    """
    preamble = preamble_override or PREAMBLE
    messages = _inject_preamble(history, preamble)

    result = agent.invoke({"messages": messages})
    lg_messages = result.get("messages", [])

    final_text = ""
    if lg_messages:
        final = lg_messages[-1]
        final_text = getattr(final, "content", "") or str(final)

    # Best-effort tool trace extraction (structures vary by version)
    tool_trace: List[Dict[str, Any]] = []
    for msg in lg_messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            tool_trace.append({"role": "assistant", "tool_calls": tool_calls})
        if getattr(msg, "tool_name", None):
            tool_trace.append(
                {
                    "role": "tool",
                    "tool_name": getattr(msg, "tool_name", ""),
                    "content": getattr(msg, "content", ""),
                }
            )

    return {"final_text": final_text, "tool_trace": tool_trace, "raw": result}
