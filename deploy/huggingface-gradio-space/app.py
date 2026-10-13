"""
Gradio app for the Llama 3 Function Agent (Hugging Face Space, free CPU tier).

Loads the fine-tuned GGUF straight from the Hub with llama-cpp-python (the same
llama.cpp engine Ollama wraps), so no Docker, Ollama or separate API server is needed.

sanitize_input / extract_json_from_response / the system prompt mirror
app/backend/main.py so behaviour matches the FastAPI version. They are copied
rather than imported because main.py pulls in FastAPI, which this Space doesn't need.
"""

import glob
import json
import os
import re

import gradio as gr
from huggingface_hub import snapshot_download
from llama_cpp import Llama

# ============================================================================
# Configuration (set GGUF_REPO as a Variable in the Space settings)
# ============================================================================

GGUF_REPO = os.environ.get("GGUF_REPO", "")
N_CTX = 4096
STOP_TOKENS = ["<|eot_id|>"]

SYSTEM_PROMPT = """You are a helpful AI assistant that can perform function calls.
When asked to perform actions, respond with a JSON object containing:
- "action": the action to perform
- "parameters": an object with relevant parameters
- "reasoning": brief explanation of your approach

Always respond with valid JSON when performing function calls."""

# ============================================================================
# Helpers (mirrors app/backend/main.py)
# ============================================================================


def sanitize_input(message: str) -> str:
    """Strip Llama 3 control tokens to prevent prompt injection, normalise whitespace."""
    dangerous_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|finetune_right_pad_id|>",
    ]
    for token in dangerous_tokens:
        message = message.replace(token, "")
    return " ".join(message.split()).strip()


def format_llama3_prompt(user_message: str) -> str:
    """Llama 3 chat format. No <|begin_of_text|>: llama.cpp adds BOS itself, and a
    manual one would give a double BOS (the model was trained with a single BOS)."""
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def extract_json_from_response(text: str):
    """Return (parsed_json | None, error | None)."""
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


# ============================================================================
# Model loading
# ============================================================================


def load_model() -> Llama:
    if not GGUF_REPO:
        raise RuntimeError(
            "Set the GGUF_REPO variable in Space settings "
            "(e.g. your-username/llama3-8b-function-calling-gguf)."
        )
    print(f"Downloading GGUF from {GGUF_REPO} ...", flush=True)
    folder = snapshot_download(
        repo_id=GGUF_REPO,
        allow_patterns=["*.gguf"],
        token=os.environ.get("HF_TOKEN"),  # only needed for private repos
    )
    gguf_files = sorted(glob.glob(f"{folder}/**/*.gguf", recursive=True))
    if not gguf_files:
        raise RuntimeError(f"No .gguf file found in {GGUF_REPO}")
    print(f"Loading {gguf_files[0]} ...", flush=True)
    return Llama(
        model_path=gguf_files[0],
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
    status = "✅ Valid JSON parsed" if error is None else f"⚠️ {error}"
    yield text, parsed, status


# ============================================================================
# UI
# ============================================================================

with gr.Blocks(title="Llama 3 Function Agent") as demo:
    gr.Markdown(
        "# 🦙 Llama 3 Function Agent\n"
        "Llama 3 8B fine-tuned with **QLoRA** for function calling "
        "(base: `unsloth/llama-3-8b-Instruct-bnb-4bit`, dataset: `glaiveai/glaive-function-calling-v2`). "
        "Runs on free CPU, so expect a few tokens per second."
    )
    with gr.Row():
        with gr.Column():
            message = gr.Textbox(
                label="Command",
                placeholder="Get the weather in Tokyo",
                lines=3,
                max_length=4096,
            )
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            max_tokens = gr.Slider(64, 1024, value=512, step=32, label="Max new tokens")
            submit = gr.Button("Run", variant="primary")
            gr.Examples(
                examples=[
                    ["Get the weather in Tokyo"],
                    ["Search the web for the latest Python release"],
                    ["Send an email to alex@example.com saying the meeting moved to 3pm"],
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
