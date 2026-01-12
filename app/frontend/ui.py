"""
Streamlit Frontend for Llama 3 Function Agent
Clean UI with Chat and JSON Output panels
"""

import json

import requests
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = "http://backend:8000"
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

    /* Light Mode Variables */
    [data-theme="light"] {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-tertiary: #e9ecef;
        --accent-cyan: #0066cc;
        --accent-magenta: #cc0066;
        --accent-green: #00aa55;
        --text-primary: #1a1a1a;
        --text-muted: #6c757d;
        --border-subtle: #dee2e6;
        --header-gradient: linear-gradient(135deg, #0066cc, #cc0066);
    }

    /* Dark Mode Variables (Default) */
    :root, [data-theme="dark"] {
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

    /* Auto-detect theme from Streamlit */
    .stApp {
        background: var(--bg-primary) !important;
    }

    [data-theme="light"] .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 50%, #ffffff 100%) !important;
    }

    [data-theme="dark"] .stApp {
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

    .json-block {
        background: var(--bg-tertiary);
        border: 1px solid var(--accent-green);
        border-radius: 6px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }

    [data-theme="light"] .json-block {
        box-shadow: 0 2px 8px rgba(0, 170, 85, 0.15);
    }

    [data-theme="dark"] .json-block {
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
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
        background: rgba(0, 170, 85, 0.1);
        border: 1px solid var(--accent-green);
        color: var(--accent-green);
    }

    [data-theme="light"] .status-online {
        background: rgba(0, 170, 85, 0.15);
    }

    .status-offline {
        background: rgba(255, 68, 68, 0.1);
        border: 1px solid #ff4444;
        color: #ff4444;
    }

    [data-theme="light"] .status-offline {
        background: rgba(255, 68, 68, 0.15);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    .status-dot.online {
        background: var(--accent-green);
    }

    [data-theme="light"] .status-dot.online {
        box-shadow: 0 0 8px rgba(0, 170, 85, 0.5);
    }

    [data-theme="dark"] .status-dot.online {
        box-shadow: 0 0 10px var(--accent-green);
    }

    .status-dot.offline {
        background: #ff4444;
        box-shadow: 0 0 10px #ff4444;
    }

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
    }

    [data-theme="light"] .stTextArea textarea:focus {
        box-shadow: 0 0 12px rgba(0, 102, 204, 0.25) !important;
    }

    [data-theme="dark"] .stTextArea textarea:focus {
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
        opacity: 0.9;
    }

    [data-theme="light"] .stButton > button:hover {
        box-shadow: 0 8px 20px rgba(0, 102, 204, 0.3) !important;
    }

    [data-theme="dark"] .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3) !important;
    }

    .sidebar .stMarkdown {
        font-family: 'Space Grotesk', sans-serif;
    }

    .error-box {
        background: rgba(255, 68, 68, 0.1);
        border: 1px solid #ff4444;
        border-radius: 8px;
        padding: 1rem;
        color: #ff4444;
        font-family: 'JetBrains Mono', monospace;
    }

    [data-theme="light"] .error-box {
        background: rgba(255, 68, 68, 0.15);
    }

    .parse-warning {
        background: rgba(255, 170, 0, 0.1);
        border: 1px solid #ffaa00;
        border-radius: 6px;
        padding: 0.75rem;
        color: #ffaa00;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    [data-theme="light"] .parse-warning {
        background: rgba(255, 170, 0, 0.15);
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
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    [data-theme="light"] .disclaimer-banner {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.2), rgba(255, 152, 0, 0.2));
        border-left-color: #f57c00;
    }

    .disclaimer-icon {
        font-size: 1.5rem;
        flex-shrink: 0;
    }

    .disclaimer-text {
        flex: 1;
        line-height: 1.5;
    }

    .footer-text {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    </style>
    <script>
    // Detect Streamlit theme and apply data-theme attribute
    (function() {
        function detectTheme() {
            const app = document.querySelector('.stApp');
            if (!app) return;
            
            const bgColor = window.getComputedStyle(app).backgroundColor;
            const rgb = bgColor.match(/\\d+/g);
            if (rgb && rgb.length >= 3) {
                const r = parseInt(rgb[0]);
                const g = parseInt(rgb[1]);
                const b = parseInt(rgb[2]);
                const isLight = (r + g + b) > 600;
                document.documentElement.setAttribute('data-theme', isLight ? 'light' : 'dark');
            }
        }
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', detectTheme);
        } else {
            detectTheme();
        }
        
        setTimeout(detectTheme, 100);
        setTimeout(detectTheme, 500);
        
        // Watch for theme changes
        const observer = new MutationObserver(detectTheme);
        observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
    })();
    </script>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helper Functions
# =============================================================================


def check_api_health() -> dict:
    """Check if the backend API and Ollama are available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "unhealthy", "ollama_connected": False, "model_available": False}
    except requests.exceptions.RequestException:
        return {"status": "unreachable", "ollama_connected": False, "model_available": False}


def send_chat_message(message: str, temperature: float = 0.7) -> dict:
    """Send a chat message to the backend API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": message, "temperature": temperature},
            timeout=120,
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        return {"success": False, "error": response.json().get("detail", "Unknown error")}
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


# =============================================================================
# Sidebar - Model Status
# =============================================================================

with st.sidebar:
    st.markdown("## 🔧 Settings")
    st.markdown("---")

    # Model Status Check
    st.markdown("### Model Status")

    if st.button("🔄 Refresh Status", use_container_width=True):
        st.session_state.health_checked = True

    health = check_api_health()

    # API Status
    if health["status"] == "healthy":
        st.markdown(
            """
            <div class="status-indicator status-online">
                <span class="status-dot online"></span>
                API Online
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-indicator status-offline">
                <span class="status-dot offline"></span>
                API Offline
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Ollama Status
    if health["ollama_connected"]:
        st.markdown(
            """
            <div class="status-indicator status-online">
                <span class="status-dot online"></span>
                Ollama Connected
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-indicator status-offline">
                <span class="status-dot offline"></span>
                Ollama Disconnected
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Model Status
    if health["model_available"]:
        st.markdown(
            """
            <div class="status-indicator status-online">
                <span class="status-dot online"></span>
                Model Ready
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-indicator status-offline">
                <span class="status-dot offline"></span>
                Model Not Loaded
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Temperature Slider
    st.markdown("### Generation Settings")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values more deterministic.",
    )

    st.markdown("---")

    # Info Section
    st.markdown("### About")
    st.markdown(
        """
        **Llama 3 Function Agent**

        This application interfaces with a
        fine-tuned Llama 3 model via Ollama
        for function calling capabilities.

        Enter natural language commands
        and receive structured JSON responses.
        """
    )


# =============================================================================
# Main Content
# =============================================================================

# Header
st.markdown('<h1 class="main-header">🦙 Llama 3 Function Agent</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Natural Language to Structured Function Calls</p>',
    unsafe_allow_html=True,
)

# Disclaimer Banner
st.markdown(
    """
    <div class="disclaimer-banner">
        <span class="disclaimer-icon">⏱️</span>
        <div class="disclaimer-text">
            <strong>Please note:</strong> Inference may take 30-120 seconds due to the hosting service's computational resources. 
            Complex queries may take longer. Please be patient while the model processes your request.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Input Section
st.markdown("### 💬 Enter Your Command")
user_input = st.text_area(
    label="Command Input",
    placeholder='Try: "Get weather in Tokyo" or "Search for Python tutorials"',
    height=100,
    label_visibility="collapsed",
)

col_btn, col_space = st.columns([1, 4])
with col_btn:
    send_button = st.button("🚀 Send", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Two Column Layout for Output
col_chat, col_json = st.columns(2)

# Initialize session state for response
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Process input
if send_button and user_input.strip():
    with st.spinner("🔄 Processing with Llama 3..."):
        result = send_chat_message(user_input.strip(), temperature)
        st.session_state.last_response = result

# Display results
with col_chat:
    st.markdown('<div class="panel-header">📝 Chat Response</div>', unsafe_allow_html=True)

    if st.session_state.last_response:
        if st.session_state.last_response["success"]:
            data = st.session_state.last_response["data"]
            response_text = data.get("response", "No response received.")

            st.markdown('<div class="output-container">', unsafe_allow_html=True)
            st.markdown(response_text)

            if data.get("tokens_used"):
                st.markdown(f"*Tokens used: {data['tokens_used']}*")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="error-box">❌ {st.session_state.last_response["error"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="output-container">', unsafe_allow_html=True)
        st.markdown("*Waiting for input...*")
        st.markdown("</div>", unsafe_allow_html=True)

with col_json:
    st.markdown('<div class="panel-header">🔧 Parsed JSON</div>', unsafe_allow_html=True)

    if st.session_state.last_response and st.session_state.last_response["success"]:
        data = st.session_state.last_response["data"]
        parsed_output = data.get("parsed_output", {})

        if parsed_output.get("parsed_json") is not None:
            json_str = format_json_output(parsed_output["parsed_json"])
            st.code(json_str, language="json")
        else:
            st.markdown('<div class="output-container">', unsafe_allow_html=True)
            st.markdown("*No valid JSON detected in response.*")

            if parsed_output.get("parse_error"):
                st.markdown(
                    f'<div class="parse-warning">⚠️ {parsed_output["parse_error"]}</div>',
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
    """
    <div class="footer-text">
        Built with FastAPI + Streamlit + Ollama | Powered by Llama 3
    </div>
    """,
    unsafe_allow_html=True,
)
