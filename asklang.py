"""
asklang.py
----------
Streamlit front-end for the LangGraph + Tavily search agent, with a polished UI.

Changes in this revision:
- The "Summarize a web page" tool is now a modal dialog opened from the sidebar.
- All URL input / summarization / results render inside the modal (no main-area output).
- WeChat-style staggered chat layout preserved (user right, assistant left).
"""

from __future__ import annotations

import os
import streamlit as st
from dotenv import load_dotenv

from graph import build_agent, invoke_agent, make_preamble
from utils.formatting import (
    collect_urls_from_langgraph_messages,
    enforce_grounded_citations,
    render_sources,
)
from utils.ui import (
    render_header,
    render_user_bubble,
    render_bot_bubble,
    render_sources_cards,
)
from utils.websum import summarize_url


# --- App bootstrap & theme ---
load_dotenv()
st.set_page_config(page_title="AskLang • LangGraph + Search", page_icon="🔎", layout="wide")

# Global CSS (scoped)
st.markdown(
    """
    <style>
      :root { --radius: 16px; --pad: 14px; --gap: 14px; }
      .main > div { padding-top: 10px; }
      section[data-testid="stSidebar"] .stButton button { width: 100%; }

      .app-title {
        font-weight: 800; letter-spacing: .2px;
        background: linear-gradient(90deg,#111827,#6b7280);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: clamp(28px, 3.2vw, 42px); margin: 6px 0 4px 0;
      }
      .subtitle { color: #6b7280; margin-bottom: 10px; }

      .chip {
        display: inline-flex; align-items: center; gap: 8px;
        background: #f3f4f6; border-radius: 999px; padding: 6px 10px;
        border: 1px solid #e5e7eb; font-size: 13px; color:#374151;
      }
      .chip.ok { border-color:#d1fae5; background:#ecfdf5; color:#065f46; }
      .chip.warn { border-color:#fee2e2; background:#fef2f2; color:#991b1b; }
      .chip .dot { width:8px; height:8px; border-radius:50%; background:#10b981; display:inline-block; }
      .chip.warn .dot { background:#ef4444; }

      .row { display: flex; width: 100%; margin: 10px 0; }
      .row.left  { justify-content: flex-start; }
      .row.right { justify-content: flex-end; }
      .row > .bubble { max-width: min(720px, 85%); }

      .bubble { border-radius: var(--radius); padding: var(--pad);
                box-shadow: 0 1px 2px rgba(0,0,0,.05); }
      .bubble.user {
        background: #111827; color: #fff;
        border-radius: 18px 6px 18px 18px;
      }
      .bubble.bot  {
        background: #fff; border: 1px solid #e5e7eb;
        border-radius: 6px 18px 18px 18px;
      }
      .bubble h4 { margin: 0 0 6px 0; font-size: 15px; color: #6b7280; font-weight: 600; }

      .sources { display: grid; grid-template-columns: repeat(auto-fill,minmax(260px,1fr)); gap: var(--gap); }
      .linkcard {
        border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px 14px;
        background: #fafafa; transition: border-color .2s ease, background .2s ease;
      }
      .linkcard:hover { border-color: #d1d5db; background: #fff; }
      .linkcard a { text-decoration: none; color: #2563eb; font-weight: 600; }
      .linkcard small { color: #6b7280; display:block; margin-top:4px; }

      div[data-testid="stChatInput"] { margin-top: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Helpers ===
has_openai = bool(os.getenv("OPENAI_API_KEY"))
has_tavily = bool(os.getenv("TAVILY_API_KEY"))

def ensure_history():
    if "history" not in st.session_state:
        st.session_state.history = []
    return st.session_state.history

def ensure_model_name() -> str:
    default_env_model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    if "model_name" not in st.session_state:
        st.session_state.model_name = default_env_model
    return st.session_state.model_name

def ensure_agent(current_model: str):
    if "agent_model_name" not in st.session_state or st.session_state.agent_model_name != current_model:
        st.session_state.agent = build_agent(model_name=current_model)
        st.session_state.agent_model_name = current_model
    return st.session_state.agent


# === Sidebar ===
with st.sidebar:
    st.header("Controls", divider="gray")

    st.markdown("**Environment**")
    c1, c2 = st.columns(2)
    with c1:  st.markdown(f'<span class="chip {"ok" if has_openai else "warn"}"><span class="dot"></span> OPENAI</span>', unsafe_allow_html=True)
    with c2:  st.markdown(f'<span class="chip {"ok" if has_tavily else "warn"}"><span class="dot"></span> TAVILY</span>', unsafe_allow_html=True)
    st.caption("Put keys in `.env` (do not commit).")

    st.markdown("**Model**")
    model_choices = ["gpt-4o-mini", "gpt-4o"]
    current_model = ensure_model_name()
    picked_model = st.selectbox(
        "OpenAI chat model",
        options=model_choices,
        index=max(0, model_choices.index(current_model)) if current_model in model_choices else 0,
        label_visibility="collapsed",
    )
    if picked_model != current_model:
        st.session_state.model_name = picked_model

    st.markdown("**Answer mode**")
    answer_mode = st.selectbox(
        "Formatting",
        options=["facts", "summary", "links"],
        index=0,
        help="Controls how the answer is formatted.",
        label_visibility="collapsed",
    )

    st.markdown("**Quick tool**")
    if st.button("Summarize a web page"):
        st.session_state["open_summary_dialog"] = True
        st.rerun()  # ensure dialog opens immediately
    
    if st.button("Summarize chat so far"):
        if st.session_state.get("history"):
            summary_prompt = "\n".join(f"{r}: {c}" for r, c in st.session_state.history)
            with st.spinner("Summarizing conversation..."):
                agent = ensure_agent(ensure_model_name())
                result = invoke_agent(agent, [{"role": "user", "content": f"Summarize this conversation:\n{summary_prompt}"}])
                st.session_state["chat_summary"] = result["final_text"]
            st.toast("✅ Summary ready! Open below.", icon="🧠")

    if "chat_summary" in st.session_state:
        with st.expander("🧠 Conversation summary"):
            st.markdown(st.session_state["chat_summary"])

    st.markdown("**Session**")
    if st.button("Clear chat history"):
        st.session_state.history = []


# === Modal dialog logic ===
def open_summary_dialog():
    """Render the URL summarization flow inside a modal dialog."""
    @st.dialog("Summarize a web page")
    def _modal():
        st.write("Paste a URL below, then click **Summarize**. The summary and sources will appear in this dialog.")
        url = st.text_input("URL", key="modal_url", placeholder="https://example.com/article")

        col_a, col_b = st.columns([1, 1])
        do_sum = col_a.button("Summarize", type="primary")
        do_close = col_b.button("Close")

        if do_sum:
            if url.strip():
                with st.spinner("Fetching & summarizing..."):
                    summary = summarize_url(url=url.strip(), model_name=ensure_model_name())
                st.success("Done.")
                st.markdown(summary)
                st.markdown(render_sources([url.strip()]))
            else:
                st.warning("Please paste a valid URL.")

        if do_close:
            st.session_state.pop("open_summary_dialog", None)
            st.rerun()

    _modal()

# Open the modal if requested
if st.session_state.pop("open_summary_dialog", False):
    open_summary_dialog()


# === Main content ===
render_header(
    title="AskLang - LangGraph Search Agent",
    subtitle="Search-augmented answers with grounded citations",
    chips=[
        ("Model", ensure_model_name()),
        ("Mode", "facts / summary / links"),
        ("Keys", f"OPENAI:{'✅' if has_openai else '❌'} TAVILY:{'✅' if has_tavily else '❌'}"),
    ],
)

history = ensure_history()
for role, content in history:
    if role == "user":
        render_user_bubble(content)
    else:
        render_bot_bubble(None, content)

prompt = st.chat_input("Type your question…")
if prompt:
    history.append(("user", prompt))
    render_user_bubble(prompt)

    agent = ensure_agent(ensure_model_name())
    preamble = make_preamble(answer_mode=answer_mode)

    result = invoke_agent(agent, history, preamble_override=preamble)
    final_text = result["final_text"] or "*No response.*"

    # Grounded citations
    raw_messages = result["raw"].get("messages", [])
    allowed_urls = collect_urls_from_langgraph_messages(raw_messages)
    grounded_links = enforce_grounded_citations(final_text, allowed_urls)

    render_bot_bubble(None, final_text)
    if grounded_links:
        render_sources_cards(grounded_links)

    history.append(("assistant", final_text))
