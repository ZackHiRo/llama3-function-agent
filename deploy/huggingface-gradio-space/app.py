"""
Gradio app for the Llama 3 Function Agent (Hugging Face Space, free CPU tier).

Loads the fine-tuned GGUF straight from the Hub with llama-cpp-python (the same
llama.cpp engine Ollama wraps), so no Docker, Ollama or separate API server is needed.

sanitize_input / extract_json_from_response / the prompt and DEFAULT_FUNCTIONS mirror
app/backend/main.py so behaviour matches the FastAPI version. They are copied
rather than imported because main.py pulls in FastAPI, which this Space doesn't need.
"""

import json
import os
import re

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ============================================================================
# Configuration (override with Variables in the Space settings)
# ============================================================================

GGUF_REPO = os.environ.get("GGUF_REPO", "AOUNZakaria/llama3-8b-function-calling-gguf")
# The repo holds two Q4_K_M files whose names differ only in case; download just one
# (the one its Modelfile uses) instead of ~10 GB.
GGUF_FILE = os.environ.get("GGUF_FILE", "llama-3-8b-Instruct.Q4_K_M.gguf")
# Local runs: point at an already-downloaded .gguf to skip the Hub download.
GGUF_PATH = os.environ.get("GGUF_PATH")
N_CTX = 4096
STOP_TOKENS = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]

# The model was fine-tuned on glaive-function-calling-v2, whose system prompt lists
# the callable functions as JSON schemas, so it is prompted the same way.
DEFAULT_FUNCTIONS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. Tokyo"},
                "units": {"type": "string", "enum": ["metric", "imperial"]},
            },
            "required": ["location"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "num_results": {"type": "integer", "description": "Number of results to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "schedule_meeting",
        "description": "Schedule a meeting in the calendar",
        "parameters": {
            "type": "object",
            "properties": {
                "participants": {"type": "array", "items": {"type": "string"}},
                "date": {"type": "string", "description": "Date of the meeting, e.g. tomorrow or 2024-06-01"},
                "time": {"type": "string", "description": "Start time, e.g. 10:00"},
                "duration_minutes": {"type": "integer"},
            },
            "required": ["participants", "date", "time"],
        },
    },
]

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required -\n" + "\n\n".join(json.dumps(fn, indent=4) for fn in DEFAULT_FUNCTIONS)
)

# ============================================================================
# Helpers (mirrors app/backend/main.py)
# ============================================================================


def sanitize_input(message: str) -> str:
    """Strip Llama 3 control tokens to prevent prompt injection, normalise whitespace."""
    message = re.sub(r"<\|[a-z0-9_]+\|>", "", message)
    lines = (" ".join(line.split()) for line in message.splitlines())
    return "\n".join(line for line in lines if line).strip()


def format_llama3_prompt(user_message: str) -> str:
    """Llama 3 chat format. No <|begin_of_text|>: llama.cpp adds BOS itself, and a
    manual one would give a double BOS (the model was trained with a single BOS)."""
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# Glaive wraps arguments in single quotes, which isn't valid JSON:
#   <functioncall> {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}
_SINGLE_QUOTED_ARGS = re.compile(r"""("arguments"\s*:\s*)'(\{.*?\})'(\s*\})""", re.DOTALL)


def _find_json_values(text: str) -> list:
    """Return every top-level JSON object/array that can be decoded from text."""
    decoder = json.JSONDecoder()
    values, i = [], 0
    while i < len(text):
        if text[i] in "{[":
            try:
                value, end = decoder.raw_decode(text, i)
                values.append(value)
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return values


def _is_function_call(value) -> bool:
    return isinstance(value, dict) and isinstance(value.get("name") or value.get("action"), str)


def extract_json_from_response(text: str):
    """Return (parsed_json | None, error | None). Single-quoted Glaive arguments are
    decoded into a nested object so the JSON view shows them structured."""
    normalized = _SINGLE_QUOTED_ARGS.sub(
        lambda m: m.group(1) + json.dumps(m.group(2)) + m.group(3), text
    )
    candidates = []
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", normalized):
        candidates.extend(_find_json_values(block))
    candidates.extend(_find_json_values(normalized))
    if not candidates:
        return None, "No valid JSON found in the response"

    parsed = next((c for c in candidates if _is_function_call(c)), candidates[0])
    if isinstance(parsed, dict) and isinstance(parsed.get("arguments"), str):
        try:
            parsed = {**parsed, "arguments": json.loads(parsed["arguments"])}
        except json.JSONDecodeError:
            pass
    return parsed, None


# ============================================================================
# Model loading
# ============================================================================


def load_model() -> Llama:
    if GGUF_PATH:
        model_path = GGUF_PATH
    else:
        print(f"Downloading {GGUF_FILE} from {GGUF_REPO} ...", flush=True)
        model_path = hf_hub_download(
            repo_id=GGUF_REPO,
            filename=GGUF_FILE,
            token=os.environ.get("HF_TOKEN"),  # only needed for private repos
        )
    print(f"Loading {model_path} ...", flush=True)
    return Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=os.cpu_count() or 2,
        verbose=False,
    )


LLM = load_model()
print("Model ready.", flush=True)

# ============================================================================
# Inference (streams tokens so the slow CPU decode is visible)
# ============================================================================


def run_agent(message: str, temperature: float, max_tokens: int):
    clean = sanitize_input(message or "")
    if not clean:
        yield "Please enter a message.", None, "⚠️ Empty message"
        return

    prompt = format_llama3_prompt(clean)
    text = ""
    for chunk in LLM.create_completion(
        prompt,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        stop=STOP_TOKENS,
        stream=True,
    ):
        text += chunk["choices"][0]["text"]
        yield text, None, "⏳ Generating..."

    parsed, error = extract_json_from_response(text)
    if _is_function_call(parsed):
        status = f"✅ Function call: `{parsed.get('name') or parsed.get('action')}`"
    elif parsed is not None:
        status = "✅ Valid JSON parsed (not a function call)"
    else:
        # Expected for out-of-scope requests: the model declines in plain text.
        status = "💬 No function call: the model answered in text"
    yield text, parsed, status


# ============================================================================
# UI
# ============================================================================

with gr.Blocks(title="Llama 3 Function Agent") as demo:
    gr.Markdown(
        "# 🦙 Llama 3 Function Agent\n"
        "Llama 3 8B fine-tuned with **QLoRA** for function calling "
        "(base: `unsloth/llama-3-8b-Instruct-bnb-4bit`, dataset: `glaiveai/glaive-function-calling-v2`). "
        "Runs on free CPU, so expect a few tokens per second.\n\n"
        "Callable functions: " + ", ".join(f"`{fn['name']}`" for fn in DEFAULT_FUNCTIONS)
    )
    with gr.Row():
        with gr.Column():
            message = gr.Textbox(
                label="Command",
                placeholder="Get the weather in Tokyo",
                lines=3,
                max_length=4096,
            )
            temperature = gr.Slider(0.0, 2.0, value=0.2, step=0.05, label="Temperature")
            max_tokens = gr.Slider(64, 1024, value=512, step=32, label="Max new tokens")
            submit = gr.Button("Run", variant="primary")
            gr.Examples(
                examples=[
                    ["Get the weather in Tokyo"],
                    ["Search the web for the latest Python release"],
                    ["Send an email to alex@example.com saying the meeting moved to 3pm"],
                    ["Schedule a 30-minute meeting with Sarah tomorrow at 10am"],
                ],
                inputs=message,
            )
        with gr.Column():
            status = gr.Markdown("")
            parsed = gr.JSON(label="Parsed JSON")
            raw = gr.Textbox(label="Raw model output", lines=8)

    outputs = [raw, parsed, status]
    submit.click(run_agent, [message, temperature, max_tokens], outputs)
    message.submit(run_agent, [message, temperature, max_tokens], outputs)

demo.queue().launch()
