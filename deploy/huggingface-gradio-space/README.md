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

## Run locally

```bash
cd deploy/huggingface-gradio-space
python -m venv .venv && source .venv/bin/activate
pip install gradio huggingface_hub -r requirements.txt   # compiles llama.cpp (a few minutes)
python app.py                                           # downloads the 4.9 GB GGUF on first run
```

Already have the GGUF? `GGUF_PATH=/path/to/model.gguf python app.py` skips the download.

Then open http://127.0.0.1:7860. On Apple Silicon, install with plain
`pip install gradio huggingface_hub llama-cpp-python` instead: the AVX flags in
`requirements.txt` are x86-only.
