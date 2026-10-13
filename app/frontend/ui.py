"""
Streamlit Frontend for Llama 3 Function Agent
Chat-style UI: natural language commands in, structured function calls out
"""

import html
import json
import os
import time

import requests
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "http://backend:8000")
CHAT_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "120"))
PAGE_TITLE = "Llama 3 Function Agent"
PAGE_ICON = "🦙"
HEALTH_TTL_SECONDS = 15

# Example requests shown in the empty state and sidebar. These are suggestions only:
# the backend's system prompt describes the JSON format but does not restrict the
# model to a fixed list of functions.
EXAMPLE_PROMPTS = [
    ("🌤️", "Weather", "What's the weather like in Tokyo right now?"),
    ("🔎", "Web search", "Search the web for the latest Python release notes"),
    ("✉️", "Email", "Send an email to alex@example.com saying the meeting moved to 3pm"),
    ("📅", "Calendar", "Schedule a 30-minute meeting with Sarah tomorrow at 10am"),
]

# Facts from notebooks/finetune.ipynb
MODEL_FACTS = [
    ("Base model", "Llama 3 8B Instruct (4-bit)"),
    ("Method", "QLoRA · r=16 · α=16"),
    ("Dataset", "Glaive function-calling v2"),
    ("Serving", "Ollama · GGUF q4_k_m"),
]

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom Styling
# =============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-tertiary: #1a1a24;
        --accent-cyan: #00d4ff;
        --accent-magenta: #ff00aa;
        --accent-green: #00ff88;
        --accent-amber: #ffaa00;
        --accent-red: #ff4d6a;
        --text-primary: #e8e8ed;
        --text-muted: #8888a0;
        --border-subtle: #2a2a3a;
        --header-gradient: linear-gradient(135deg, #00d4ff, #ff00aa);
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(0, 212, 255, 0.08), transparent 60%),
            radial-gradient(900px 500px at 110% 10%, rgba(255, 0, 170, 0.07), transparent 60%),
            var(--bg-primary) !important;
    }

    html, body, [class*="css"], .stMarkdown, .stButton button {
        font-family: 'Space Grotesk', sans-serif;
    }

    .block-container { max-width: 980px; padding-top: 2rem; }

    /* ---------- Header ---------- */
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: var(--header-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }

    .sub-header {
        color: var(--text-muted);
        text-align: center;
        font-size: 1rem;
        margin-bottom: 1.75rem;
    }

    /* ---------- Empty state ---------- */
    .hero {
        text-align: center;
        padding: 1.5rem 1rem 0.5rem;
        color: var(--text-muted);
    }
    .hero h3 { color: var(--text-primary); font-weight: 600; margin-bottom: 0.35rem; }

    /* ---------- Buttons ---------- */
    .stButton > button {
        font-weight: 500 !important;
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        text-align: left !important;
        transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stButton > button:hover {
        border-color: var(--accent-cyan) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.15) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--header-gradient) !important;
        border: none !important;
        color: white !important;
    }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {
        background: rgba(18, 18, 26, 0.7);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
        animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stChatInput"] textarea {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ---------- Function call card ---------- */
    .fn-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-muted);
        margin-bottom: 0.2rem;
    }
    .fn-name {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--accent-cyan);
        margin-bottom: 0.6rem;
    }
    .fn-name .paren { color: var(--text-muted); font-weight: 400; }
    .fn-reasoning {
        border-left: 3px solid var(--accent-magenta);
        padding: 0.35rem 0.75rem;
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* ---------- Badges ---------- */
    .badge-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.75rem; }
    .badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        border: 1px solid var(--border-subtle);
        color: var(--text-muted);
        background: var(--bg-tertiary);
    }
    .badge-ok { color: var(--accent-green); border-color: rgba(0, 255, 136, 0.4); background: rgba(0, 255, 136, 0.08); }
    .badge-warn { color: var(--accent-amber); border-color: rgba(255, 170, 0, 0.4); background: rgba(255, 170, 0, 0.08); }

    /* ---------- Sidebar status ---------- */
    .status-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 0.4rem 0.9rem;
        margin-bottom: 0.75rem;
    }
    .status-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.45rem 0;
        font-size: 0.9rem;
    }
    .status-row + .status-row { border-top: 1px solid var(--border-subtle); }
    .status-label { flex: 1; color: var(--text-primary); }
    .status-value { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; }
    .status-value.online { color: var(--accent-green); }
    .status-value.offline { color: var(--accent-red); }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; animation: pulse 2s infinite; }
    .status-dot.online { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
    .status-dot.offline { background: var(--accent-red); box-shadow: 0 0 8px var(--accent-red); }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }

    .fact-row { display: flex; justify-content: space-between; gap: 0.5rem; font-size: 0.82rem; padding: 0.2rem 0; }
    .fact-key { color: var(--text-muted); }
    .fact-val { color: var(--text-primary); text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helper Functions
# =============================================================================


@st.cache_data(ttl=HEALTH_TTL_SECONDS, show_spinner=False)
def check_api_health() -> dict:
    """Check if the backend API and Ollama are available.

    Cached for a few seconds so ordinary interactions (sliders, buttons) don't
    each wait on a network round-trip.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "unhealthy", "ollama_connected": False, "model_available": False}
    except requests.exceptions.RequestException:
        return {"status": "unreachable", "ollama_connected": False, "model_available": False}


def send_chat_message(message: str, temperature: float = 0.7, max_tokens: int = 512) -> dict:
    """Send a chat message to the backend API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": message, "temperature": temperature, "max_tokens": max_tokens},
            timeout=CHAT_TIMEOUT,
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        try:
            detail = response.json().get("detail", "Unknown error")
        except ValueError:
            detail = f"HTTP {response.status_code}"
        return {"success": False, "error": str(detail)}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. The model may be processing a complex query."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to the backend API."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def format_json_output(data: dict | list | None) -> str:
    """Format JSON data for display."""
    if data is None:
        return "null"
    return json.dumps(data, indent=2, ensure_ascii=False)


def badge(label: str, tone: str = "neutral") -> str:
    return f'<span class="badge badge-{tone}">{html.escape(label)}</span>'


def extract_function_call(parsed: object) -> tuple[str | None, object, str | None]:
    """Return (function name, arguments, reasoning) from parsed model JSON.

    Supports the app's prompt format ({"action", "parameters", "reasoning"}) and the
    Glaive training format ({"name", "arguments"}), where arguments may be a JSON string.
    """
    if not isinstance(parsed, dict):
        return None, None, None
    name = parsed.get("action") or parsed.get("name")
    args = parsed.get("parameters", parsed.get("arguments"))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    reasoning = parsed.get("reasoning")
    return (str(name) if name else None), args, (str(reasoning) if reasoning else None)


def queue_prompt(prompt: str) -> None:
    st.session_state.queued_prompt = prompt


def clear_history() -> None:
    st.session_state.history = []


def render_assistant_turn(turn: dict) -> None:
    """Render one model reply: badges, function-call card, raw output."""
    if not turn["ok"]:
        st.error(f"❌ {turn['error']}")
        return

    data = turn["data"] or {}
    parsed_output = data.get("parsed_output") or {}
    parsed_json = parsed_output.get("parsed_json")
    raw_text = data.get("response", "")

    badges = [
        badge("✓ valid JSON", "ok") if parsed_json is not None else badge("✗ no valid JSON", "warn"),
        badge(f"⏱ {turn['latency']:.1f}s"),
    ]
    if data.get("tokens_used"):
        badges.append(badge(f"{data['tokens_used']} tokens"))
    if data.get("model"):
        badges.append(badge(data["model"]))
    st.markdown(f'<div class="badge-row">{"".join(badges)}</div>', unsafe_allow_html=True)

    name, args, reasoning = extract_function_call(parsed_json)
    if name:
        with st.container(border=True):
            st.markdown(
                f'<div class="fn-label">Function call</div>'
                f'<div class="fn-name">{html.escape(name)}<span class="paren">( )</span></div>',
                unsafe_allow_html=True,
            )
            if args not in (None, "", {}):
                st.code(format_json_output(args) if not isinstance(args, str) else args, language="json")
            if reasoning:
                st.markdown(f'<div class="fn-reasoning">{html.escape(reasoning)}</div>', unsafe_allow_html=True)
        with st.expander("Full JSON"):
            st.code(format_json_output(parsed_json), language="json")
    elif parsed_json is not None:
        st.code(format_json_output(parsed_json), language="json")
    else:
        st.markdown(raw_text or "_Empty response._")
        if parsed_output.get("parse_error"):
            st.caption(f"⚠️ {parsed_output['parse_error']}")

    with st.expander("Raw model output"):
        st.code(raw_text, language="text")


# =============================================================================
# Session State
# =============================================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("### ⚡ System status")

    health = check_api_health()
    rows = [
        ("API", health["status"] == "healthy", "online", "offline"),
        ("Ollama", health["ollama_connected"], "connected", "down"),
        ("Model", health["model_available"], "ready", "not loaded"),
    ]
    status_html = "".join(
        f'<div class="status-row">'
        f'<span class="status-dot {"online" if ok else "offline"}"></span>'
        f'<span class="status-label">{label}</span>'
        f'<span class="status-value {"online" if ok else "offline"}">{ok_text if ok else bad_text}</span>'
        f"</div>"
        for label, ok, ok_text, bad_text in rows
    )
    st.markdown(f'<div class="status-card">{status_html}</div>', unsafe_allow_html=True)

    if st.button("🔄 Refresh status", use_container_width=True):
        check_api_health.clear()
        st.rerun()

    st.markdown("### 🎛️ Generation")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.2,
        step=0.1,
        help="Lower values give more deterministic, schema-following output.",
    )
    max_tokens = st.slider(
        "Max new tokens",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
        help="Function calls are short; a lower limit returns faster on CPU.",
    )

    st.markdown("### 🧰 Try a function")
    for icon, title, prompt in EXAMPLE_PROMPTS:
        st.button(
            f"{icon}  {title}",
            key=f"side-{title}",
            use_container_width=True,
            on_click=queue_prompt,
            args=(prompt,),
            help=prompt,
        )

    st.markdown("### 🦙 Model")
    facts_html = "".join(
        f'<div class="fact-row"><span class="fact-key">{k}</span><span class="fact-val">{v}</span></div>'
        for k, v in MODEL_FACTS
    )
    st.markdown(facts_html, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("---")
        st.download_button(
            "⬇️ Export session (JSON)",
            data=json.dumps(st.session_state.history, indent=2, ensure_ascii=False),
            file_name="function_agent_session.json",
            mime="application/json",
            use_container_width=True,
        )
        st.button("🗑️ Clear conversation", use_container_width=True, on_click=clear_history)

# =============================================================================
# Main Content
# =============================================================================

st.markdown('<h1 class="main-header">🦙 Llama 3 Function Agent</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Natural language in → structured function calls out</p>',
    unsafe_allow_html=True,
)

if not health["model_available"]:
    st.warning(
        "The model isn't reported as ready yet. If you're running locally, start Ollama and run "
        "`ollama create llama3-function-calling -f Modelfile`, then refresh the status.",
        icon="⚠️",
    )

# chat_input always renders at the bottom of the page, wherever it's called.
typed_prompt = st.chat_input("Describe an action, e.g. “Get the weather in Tokyo”", max_chars=4096)
prompt = typed_prompt or st.session_state.pop("queued_prompt", None)

if not st.session_state.history and not prompt:
    st.markdown(
        '<div class="hero"><h3>What should the agent do?</h3>'
        "Describe an action in plain English and the fine-tuned model turns it into a JSON function call.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    cols = st.columns(2)
    for i, (icon, title, example) in enumerate(EXAMPLE_PROMPTS):
        with cols[i % 2]:
            st.button(
                f"{icon}  **{title}**  \n{example}",
                key=f"hero-{title}",
                use_container_width=True,
                on_click=queue_prompt,
                args=(example,),
            )

for turn in st.session_state.history:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(turn["prompt"])
    with st.chat_message("assistant", avatar="🦙"):
        render_assistant_turn(turn)

if prompt:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🦙"):
        with st.spinner("Working out the function call… (CPU inference can take 30s or more)"):
            started = time.perf_counter()
            result = send_chat_message(prompt, temperature, max_tokens)
            latency = time.perf_counter() - started
        turn = {
            "prompt": prompt,
            "ok": result["success"],
            "data": result.get("data"),
            "error": result.get("error"),
            "latency": latency,
        }
        render_assistant_turn(turn)
    st.session_state.history.append(turn)

if st.session_state.history:
    st.caption(
        "Each command is handled independently: the model doesn't see earlier messages in this conversation."
    )
