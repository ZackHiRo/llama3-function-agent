---
title: Llama 3 Function Agent
emoji: 🦙
colorFrom: indigo
colorTo: purple
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
---

# 🦙 Llama 3 Function Agent

Llama 3 8B fine-tuned for function calling with QLoRA, served from a GGUF via llama-cpp-python.
By default it loads `llama-3-8b-Instruct.Q4_K_M.gguf` from
[AOUNZakaria/llama3-8b-function-calling-gguf](https://huggingface.co/AOUNZakaria/llama3-8b-function-calling-gguf);
override with the `GGUF_REPO` / `GGUF_FILE` variables in Settings → Variables and secrets.
CPU inference is slow: expect a few tokens per second.
