---
title: Llama 3 Function Agent
emoji: 🦙
colorFrom: cyan
colorTo: pink
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
license: mit
---

# 🦙 Llama 3 Function Agent

A fine-tuned Llama 3 8B model for structured function calling — converts natural language commands into structured JSON.

## Usage

Type any natural language command and the model will respond with a structured JSON function call:

```
"Get weather in Tokyo"
→ {"action": "get_weather", "parameters": {"location": "Tokyo"}, "reasoning": "..."}
```

## Setup (Space Secrets)

Set the following secrets in your Space settings:

| Secret | Description |
|--------|-------------|
| `HF_MODEL_ID` | Your model repo, e.g. `username/llama3-function-calling` |
| `HF_TOKEN` | HuggingFace token (only needed for private models) |

## Model

Fine-tuned on [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) using QLoRA + Unsloth.
