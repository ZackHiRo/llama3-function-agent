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
Set the `GGUF_REPO` variable in Settings → Variables and secrets.
CPU inference is slow: expect a few tokens per second.
