"""
Llama 3 Function Agent — HuggingFace Spaces Edition
Streamlit app that calls your fine-tuned model via the HF Inference API.
Set HF_MODEL_ID and (optionally) HF_TOKEN as Space secrets.
"""

import json
import os
import re

import requests
import streamlit as st

# =============================================================================
# Configuration — edit HF_MODEL_ID or set it as a Space secret
# =============================================================================

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "YOUR_HF_USERNAME/YOUR_MODEL_NAME")
HF_API_TOKEN = os.environ.get("HF_TOKEN", "")  # optional for public models
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
REQUEST_TIMEOUT = 180  # seconds — cold-start can be slow on free tier

PAGE_TITLE = "Llama 3 Function Agent"
PAGE_ICON = "🦙"

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
        --text-primary: #e8e8ed;
        --text-muted: #8888a0;
        --border-subtle: #2a2a3a;
        --header-gradient: linear-gradient(135deg, #00d4ff, #ff00aa);
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d0d14 50%, #0a0a0f 100%) !important;
    }

    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--header-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .sub-header {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-muted);
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .panel-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent-cyan);
        padding: 0.75rem 1rem;
        background: var(--bg-tertiary);
        border-radius: 8px 8px 0 0;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 0;
    }

    .output-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 0 0 8px 8px;
        padding: 1.25rem;
        min-height: 200px;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-primary);
    }

    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .status-online {
        background: rgba(0, 255, 136, 0.15);
        border: 1px solid var(--accent-green);
        color: var(--accent-green);
    }

    .status-offline {
        background: rgba(255, 68, 68, 0.15);
        border: 1px solid #ff4444;
        color: #ff4444;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    .status-dot.online { background: var(--accent-green); box-shadow: 0 0 10px var(--accent-green); }
    .status-dot.offline { background: #ff4444; box-shadow: 0 0 10px #ff4444; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.2) !important;
    }

    .stButton > button {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        background: var(--header-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3) !important;
    }

    .error-box {
        background: rgba(255, 68, 68, 0.15);
        border: 1px solid #ff4444;
        border-radius: 8px;
        padding: 1rem;
        color: #ff4444;
        font-family: 'JetBrains Mono', monospace;
    }

    .parse-warning {
        background: rgba(255, 170, 0, 0.15);
        border: 1px solid #ffaa00;
        border-radius: 6px;
        padding: 0.75rem;
        color: #ffaa00;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    .disclaimer-banner {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.15), rgba(255, 152, 0, 0.15));
        border-left: 4px solid #ff9800;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.9rem;
        color: var(--text-primary);
    }

    .footer-text {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Inference Logic (replaces FastAPI backend + Ollama)
# =============================================================================

SYSTEM_PROMPT = """You are a helpful AI assistant that can perform function calls.
When asked to perform actions, respond with a JSON object containing:
- "action": the action to perform
- "parameters": an object with relevant parameters
- "reasoning": brief explanation of your approach

Always respond with valid JSON when performing function calls."""


def format_llama3_prompt(user_message: str) -> str:
    """Format message into Llama 3 ChatML template."""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def sanitize_input(message: str) -> str:
    """Remove Llama special tokens to prevent prompt injection."""
    dangerous = [
        "<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>",
        "<|end_header_id|>", "<|eot_id|>", "<|finetune_right_pad_id|>",
    ]
    for token in dangerous:
        message = message.replace(token, "")
    return " ".join(message.split()).strip()


def extract_json(text: str) -> tuple:
    """Try to extract a JSON object/array from raw model text."""
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"\{[\s\S]*\}",
        r"\[[\s\S]*\]",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            try:
                return json.loads(match), None
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text.strip()), None
    except json.JSONDecodeError as e:
        return None, f"Could not parse JSON: {e}"


def call_hf_inference(prompt: str, temperature: float, max_tokens: int) -> tuple:
    """
    Call the HuggingFace Inference API.
    Returns (response_text, error_message).
    """
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "stop": ["<|eot_id|>", "<|end_of_text|>"],
        },
    }

    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip(), None
        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            if "loading" in err.lower():
                return "", "Model is loading on HuggingFace servers — please wait ~20 seconds and try again."
            return "", f"HuggingFace API error: {err}"

        return str(data), None

    except requests.exceptions.Timeout:
        return "", "Request timed out. The model may be cold-starting — try again in a moment."
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 503:
            return "", "Model is loading on HuggingFace servers — please try again in ~20 seconds."
        return "", f"HTTP error {e.response.status_code}: {e.response.text}"
    except requests.exceptions.RequestException as e:
        return "", f"Connection error: {e}"


def check_model_status() -> tuple:
    """Ping HF API to see if model is reachable."""
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    try:
        r = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": "ping", "parameters": {"max_new_tokens": 1}},
            timeout=10,
        )
        if r.status_code == 200:
            return True, "Model Ready"
        if r.headers.get("content-type", "").startswith("application/json"):
            data = r.json()
            if "loading" in data.get("error", "").lower():
                return False, "Model Loading..."
        return False, f"Status {r.status_code}"
    except Exception:
        return False, "Unreachable"


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")
    st.markdown("### Model Status")

    if st.button("Refresh Status", use_container_width=True):
        st.rerun()

    model_ok, model_msg = check_model_status()
    cls = "online" if model_ok else "offline"
    st.markdown(
        f'<div class="status-indicator status-{cls}">'
        f'<span class="status-dot {cls}"></span>'
        f'{model_msg}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    hf_model_display = (
        HF_MODEL_ID if HF_MODEL_ID != "YOUR_HF_USERNAME/YOUR_MODEL_NAME" else "Not configured"
    )
    st.markdown(f"**Model:** `{hf_model_display}`")

    st.markdown("---")
    st.markdown("### Generation Settings")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more deterministic.",
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
        help="Maximum tokens in the model response.",
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """
        **Llama 3 Function Agent**

        Fine-tuned on the Glaive Function Calling
        dataset. Enter natural language commands
        and receive structured JSON responses.
        """
    )


# =============================================================================
# Main Content
# =============================================================================

st.markdown('<h1 class="main-header">Llama 3 Function Agent</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Natural Language to Structured Function Calls</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-banner">
        Note: Inference may take 30-120 seconds on the free HuggingFace tier.
        If the model is cold-starting, simply wait 20 seconds and retry.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Enter Your Command")
user_input = st.text_area(
    label="Command Input",
    placeholder='Try: "Get weather in Tokyo" or "Search for Python tutorials"',
    height=100,
    label_visibility="collapsed",
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    send_button = st.button("Send", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Session state
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Process
if send_button and user_input.strip():
    clean = sanitize_input(user_input.strip())
    if not clean:
        st.error("Message is empty after sanitization.")
    else:
        with st.spinner("Calling HuggingFace Inference API..."):
            prompt = format_llama3_prompt(clean)
            raw_text, error = call_hf_inference(prompt, temperature, max_tokens)

        if error:
            st.session_state.last_response = {"success": False, "error": error}
        else:
            parsed_json, parse_error = extract_json(raw_text)
            st.session_state.last_response = {
                "success": True,
                "raw_text": raw_text,
                "parsed_json": parsed_json,
                "parse_error": parse_error,
            }

# Display results
col_chat, col_json = st.columns(2)

with col_chat:
    st.markdown('<div class="panel-header">Chat Response</div>', unsafe_allow_html=True)

    if st.session_state.last_response:
        resp = st.session_state.last_response
        if resp["success"]:
            st.markdown('<div class="output-container">', unsafe_allow_html=True)
            st.markdown(resp["raw_text"] or "*Empty response from model.*")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="error-box">{resp["error"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="output-container">', unsafe_allow_html=True)
        st.markdown("*Waiting for input...*")
        st.markdown("</div>", unsafe_allow_html=True)

with col_json:
    st.markdown('<div class="panel-header">Parsed JSON</div>', unsafe_allow_html=True)

    if st.session_state.last_response and st.session_state.last_response["success"]:
        resp = st.session_state.last_response
        if resp.get("parsed_json") is not None:
            st.code(json.dumps(resp["parsed_json"], indent=2, ensure_ascii=False), language="json")
        else:
            st.markdown('<div class="output-container">', unsafe_allow_html=True)
            st.markdown("*No valid JSON detected in response.*")
            if resp.get("parse_error"):
                st.markdown(
                    f'<div class="parse-warning">{resp["parse_error"]}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="output-container">', unsafe_allow_html=True)
        st.markdown("*JSON output will appear here...*")
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown(
    '<div class="footer-text">Built with Streamlit · Powered by Llama 3 on HuggingFace</div>',
    unsafe_allow_html=True,
)
