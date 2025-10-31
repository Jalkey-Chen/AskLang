from __future__ import annotations
import urllib.parse
from html import escape as html_escape
import streamlit as st


def _favicon(url: str) -> str:
    """Return a small site favicon URL via Google's favicon service."""
    try:
        netloc = urllib.parse.urlparse(url).netloc
        domain = netloc.split(":")[0]
        return f"https://www.google.com/s2/favicons?domain={domain}"
    except Exception:
        return "https://www.google.com/s2/favicons?domain=example.com"


def render_header(title: str, subtitle: str, chips: list[tuple[str, str]]):
    """Render a header with gradient title and info chips."""
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)

    cols = st.columns(len(chips))
    for (label, value), c in zip(chips, cols):
        c.markdown(f'<span class="chip"><b>{label}</b> · {value}</span>', unsafe_allow_html=True)
    st.divider()


def _escape_user_text(text: str) -> str:
    """Escape user text for safe HTML embedding (no Markdown in user bubble)."""
    if text is None:
        return ""
    return html_escape(text).replace("\n", "<br/>")


def render_user_bubble(text: str):
    """Right-aligned dark bubble for the user (WeChat-style)."""
    safe = _escape_user_text(text)
    st.markdown(f'<div class="row right"><div class="bubble user">{safe}</div></div>', unsafe_allow_html=True)


def render_bot_bubble(title: str | None, text: str):
    """Left-aligned light bubble for the assistant (WeChat-style)."""
    if title:
        st.markdown(f'<div class="row left"><div class="bubble bot"><h4>{title}</h4>{text}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="row left"><div class="bubble bot">{text}</div></div>', unsafe_allow_html=True)


def _hostname(url: str) -> str:
    """Extract a pretty host label for a URL."""
    try:
        host = urllib.parse.urlparse(url).netloc
        return host.replace("www.", "")
    except Exception:
        return url


def render_sources_cards(urls: list[str]):
    """Render a grid of link cards for Sources."""
    if not urls:
        return
    st.markdown('<div class="sources">', unsafe_allow_html=True)
    for u in urls:
        host = _hostname(u)
        fav = _favicon(u)
        st.markdown(
            f'''
            <div class="linkcard">
              <img src="{fav}" width="16" height="16" style="vertical-align:middle; margin-right:6px;" />
              <a href="{u}" target="_blank">{host}</a>
              <small>{u}</small>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)
