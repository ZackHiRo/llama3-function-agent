---
title: Llama 3 Function Agent
emoji: 🦙
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🦙 Llama 3 Function Agent

Streamlit UI + FastAPI backend + Ollama, all in one container, serving a Llama 3 8B model
fine-tuned for function calling with QLoRA.

On first boot the container downloads the GGUF from the Hugging Face Hub (default
`AOUNZakaria/llama3-8b-function-calling-gguf`, override with the `GGUF_REPO` / `GGUF_FILE` variables),
registers it with Ollama, and the UI switches from "Model Not Loaded" to ready.
CPU inference is slow (a few tokens/second): the first reply can take a minute or more.
