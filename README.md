
# 🔎 AskLang — LangGraph Search Agent

A minimal, **search-augmented** chat app built with **LangGraph + LangChain**, **OpenAI**, and **Tavily**.  
It answers questions with **grounded citations**, offers a **web-page summarizer** in a modal dialog, and provides a clean **WeChat-style** chat UI.

---

## ✨ Features

- **Web search tool (Tavily)** with grounded citations  
  – Sources are restricted to the URLs returned by the current search run.

- **Answer modes**: `facts` · `summary` · `links`  
  – Switch in the sidebar; the system preamble adapts.

- **Web page summarizer (modal)**  
  – Click **“Summarize a web page (open dialog)”** in the sidebar, paste a URL, and get a short summary with sources **inside the dialog** (no main-area clutter).

- **Model picker** (sidebar)  
  – Choose between `gpt-4o-mini` and `gpt-4o` at runtime.

- **Polished UI**  
  – WeChat-style left/right chat bubbles, link cards for sources, and a compact sidebar.

---

## 🗂️ Repository Structure

```

.
├── asklang.py                 # Streamlit app (UI + interactions)
├── graph.py                   # Agent builder & invoke helpers (LangGraph + tools)
├── utils/
│   ├── formatting.py          # URL extraction, grounded-citation helpers, renderers
│   ├── ui.py                  # Reusable UI components (bubbles, cards, header)
│   └── websum.py              # Web page fetch + LLM summarization
└── .streamlit/
└── config.toml            # Theme (optional)

````

---

## ⚙️ Requirements

- Python **3.10+**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Streamlit **1.31+** (for `st.dialog`).  
  If you’re on an older Streamlit, change `@st.dialog` → `@st.experimental_dialog`.

---

## 🔑 Environment Variables

Create a `.env` file in the project root (do **not** commit it):

**Where to get keys**

* **OpenAI**: from your account’s API Keys page.
* **Tavily**: create an account at tavily.com and generate a key on the dashboard.

---

## 🚀 Quickstart

```bash
# 1) Clone the repo
git clone <your-repo-url>
cd <your-repo-folder>

# 2) Install deps (from pyproject) with uv
uv sync

# 3) Create .env (see template above) and fill your keys
#    OPENAI_API_KEY and TAVILY_API_KEY are required.

# 4) Run the app
uv run streamlit run asklang.py
```

> No `pyproject.toml` yet? You can add deps manually:
>
> ```bash
> uv add streamlit python-dotenv langchain langgraph langchain-openai \
>        langchain-community langchain-tavily tavily-python
> ```

---

## 🧭 How to Use

1. **Ask a question** in the chat input (e.g., “Who is the current quarterback for the Chicago Bears?”).
   The agent will call Tavily as needed and show a short answer with grounded **Sources**.

2. **Switch Answer mode** in the sidebar:

   * `facts`: short bullet/line + sources
   * `summary`: concise paragraph + sources
   * `links`: one-liner + a compact list of sources

3. **Summarize a web page**

   * Sidebar → **“Summarize a web page (open dialog)”** → paste URL → **Summarize**.
   * The summary and sources appear **inside the dialog**.

4. **Change model** anytime via the sidebar select box.

---

## 🧠 Architecture

* **LLM**: `langchain_openai.ChatOpenAI`.
* **Search Tool**: `langchain_tavily.TavilySearch` (falls back to the legacy community tool if needed).
* **Agent**: LangGraph’s `create_react_agent(model, [search])`.
* **Grounded citations**:
  We extract URLs from this run’s tool outputs and **filter** the model’s sources so only those URLs appear. If the model didn’t include links, we fall back to the top Tavily URLs.

Key entry points:

* `graph.build_agent(model_name)` – constructs the agent with tools.
* `graph.invoke_agent(agent, history, preamble_override)` – runs the graph and normalizes the response.
* `utils/formatting.enforce_grounded_citations(answer_text, allowed_urls)` – URL grounding.
* `utils/websum.summarize_url(url, model_name)` – fetch + LLM summary for the modal tool.

---

## 🧪 Troubleshooting

* **`MissingAPIKeyError: No API key provided`**
  → Ensure `.env` is present and `OPENAI_API_KEY` + `TAVILY_API_KEY` are set. Restart the app.

* **`TypeError: create_react_agent() missing 1 required positional argument: 'model'`**
  → We call `create_react_agent(model, tools)` and fall back to `create_react_agent(llm=..., tools=...)` for older versions.
  If you still see it, update `langgraph` / `langchain` or delete cached venv and `uv sync` again.

* **Tavily deprecation warnings** (`TavilySearchResults` deprecated)
  → This project uses **`langchain-tavily`**. If you cannot install it, the code will use the legacy community tool as a fallback.

* **Dialog keeps popping up when sending a message**
  → We now use a **one-shot** session flag (`st.session_state.pop("open_summary_dialog", False)`), so the dialog won’t re-open automatically after you close it.

* **Dialog not available**
  → If your Streamlit is older, replace `@st.dialog` with `@st.experimental_dialog`.

---

## 📜 License

MIT

---

## 🙏 Credits

* [Streamlit](https://streamlit.io/)
* [LangChain](https://python.langchain.com/)
* [LangGraph](https://langchain-ai.github.io/langgraph/)
* [Tavily](https://tavily.com/)
* OpenAI API
