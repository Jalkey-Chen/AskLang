"""
utils/formatting.py
-------------------
Small formatting and citation helpers for the Streamlit UI.

New helpers for A (Grounded Citations):
- collect_urls_from_langgraph_messages(messages): best-effort extraction
  of URLs from tool outputs (e.g., Tavily JSON).
- enforce_grounded_citations(answer_text, allowed_urls): restrict the final
  Sources to URLs that appeared in the current tool results.
"""

from __future__ import annotations

import re
from typing import List, Iterable, Any


_URL_RE = re.compile(
    r"(https?://[^\s)]+)",
    flags=re.IGNORECASE,
)


def extract_urls(text: str) -> List[str]:
    """
    Extract all HTTP/HTTPS URLs from a text blob.

    Parameters
    ----------
    text : str
        The input content where URLs might appear.

    Returns
    -------
    list[str]
        Unique URLs in the order they first appear.
    """
    if not text:
        return []
    seen = set()
    urls = []
    for m in _URL_RE.finditer(text):
        url = m.group(1).rstrip(".,);]}")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def render_sources(links: List[str]) -> str:
    """
    Render a list of source links as a markdown bullet list.

    Parameters
    ----------
    links : list[str]
        Source URLs.

    Returns
    -------
    str
        Markdown-formatted bullet list.
    """
    if not links:
        return "_No sources._"
    return "\n".join(f"- {u}" for u in links)


def _flatten(obj: Any) -> str:
    """Return a plain string representation of an arbitrary object."""
    try:
        if isinstance(obj, str):
            return obj
        return str(obj)
    except Exception:
        return ""


def collect_urls_from_langgraph_messages(messages: Iterable[Any]) -> List[str]:
    """
    Best-effort collection of URLs from LangGraph's intermediate messages.
    This is intentionally permissive: we regex-scan the 'content' of any
    message that looks like tool output (e.g., from Tavily).

    Parameters
    ----------
    messages : iterable[Any]
        The raw `result["messages"]` list returned by the agent.

    Returns
    -------
    list[str]
        Unique URLs discovered in tool outputs (e.g., Tavily result JSON).
    """
    seen = set()
    out: List[str] = []

    for msg in messages or []:
        # Try common fields across message variants
        tool_name = getattr(msg, "tool_name", "") or getattr(msg, "name", "")
        content = getattr(msg, "content", "")
        if not content:
            content = _flatten(msg)

        # Only scan likely tool outputs OR scan everything if uncertain.
        # We bias toward names that include 'tavily' or 'search'.
        if tool_name:
            name_l = str(tool_name).lower()
            if "tavily" in name_l or "search" in name_l or "tool" in name_l:
                for u in extract_urls(str(content)):
                    if u not in seen:
                        seen.add(u)
                        out.append(u)
        else:
            # Fallback: scan any message whose content looks JSON-ish or contains URLs.
            if "http" in str(content):
                for u in extract_urls(str(content)):
                    if u not in seen:
                        seen.add(u)
                        out.append(u)

    return out


def enforce_grounded_citations(answer_text: str, allowed_urls: List[str]) -> List[str]:
    """
    Restrict final sources to URLs that appeared in the *current* tool outputs.

    Parameters
    ----------
    answer_text : str
        The LLM's final answer (may contain URLs).
    allowed_urls : list[str]
        URLs extracted from this run's tool messages (Tavily results).

    Returns
    -------
    list[str]
        Final list of grounded URLs to display. If the answer contains no URLs
        that match the allowed set, we return the first few from `allowed_urls`.
    """
    if not allowed_urls:
        return extract_urls(answer_text)

    answer_urls = extract_urls(answer_text)
    allowed_set = {u.split("#")[0] for u in allowed_urls}

    grounded = []
    for u in answer_urls:
        base = u.split("#")[0]
        if base in allowed_set and base not in grounded:
            grounded.append(base)

    if grounded:
        return grounded

    # Fallback: if the answer had no valid URLs, show top-k allowed ones.
    return list(allowed_set)[:3]
