"""
utils/websum.py
---------------
Minimal web page summarization utility used by the Streamlit sidebar tool.

Pipeline
1) Fetch the page with WebBaseLoader.
2) Concatenate texts (trim to a safe size).
3) Ask the LLM for a concise summary with 3–5 key points.

Notes
- This intentionally avoids advanced parsing; for production you may add
  HTML -> readability extraction, boilerplate removal, and deduplication.
"""

from __future__ import annotations

from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader


def _load_url_text(url: str, max_chars: int = 12000) -> str:
    """
    Load a web page and return a plain text blob (trimmed).

    Parameters
    ----------
    url : str
        The URL to load.
    max_chars : int
        Hard cap on characters fed to the LLM to control cost and latency.

    Returns
    -------
    str
        Concatenated page text (trimmed).
    """
    docs = WebBaseLoader(url).load()
    text = "\n\n".join(d.page_content for d in docs if d.page_content)
    return text[:max_chars]


def summarize_url(url: str, model_name: str = "gpt-4o-mini") -> str:
    """
    Fetch `url` and produce a short, citation-ready summary.

    Parameters
    ----------
    url : str
        The web page URL to summarize.
    model_name : str
        OpenAI chat model to use.

    Returns
    -------
    str
        A concise markdown summary (bulleted) suitable for UI display.
    """
    text = _load_url_text(url)
    if not text.strip():
        return "_Failed to fetch or parse content from the URL._"

    llm = ChatOpenAI(model=model_name, temperature=0)
    prompt = (
        "You are a concise summarizer.\n"
        "Summarize the following web page into 3–5 bullet points. "
        "Capture key facts, dates, numbers, and named entities. "
        "Avoid speculation. Keep total length within ~120 words.\n\n"
        f"=== PAGE TEXT START ===\n{text}\n=== PAGE TEXT END ==="
    )
    resp = llm.invoke(prompt)
    content = getattr(resp, "content", None) or str(resp)
    return content
