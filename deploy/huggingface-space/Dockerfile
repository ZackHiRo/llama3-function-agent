# =============================================================================
# Llama 3 Function Agent - Hugging Face Space (Docker SDK)
# =============================================================================
# One container runs: Ollama (11434) + FastAPI backend (8000) + Streamlit UI (7860).
# Hugging Face only exposes the port declared as `app_port` in README.md (7860).
# Build context is the Space repo root, which must contain app/, start.sh, README.md.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# zstd is required by the Ollama installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

# Hugging Face Spaces run containers as UID 1000
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/home/user/.ollama/models \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    OLLAMA_MODEL=llama3-function-calling \
    API_BASE_URL=http://127.0.0.1:8000 \
    REQUEST_TIMEOUT=600

WORKDIR /home/user/app

# Dependencies first so code changes don't invalidate this layer
COPY --chown=user app/backend/requirements.txt backend-requirements.txt
COPY --chown=user app/frontend/requirements.txt frontend-requirements.txt
RUN pip install -r backend-requirements.txt -r frontend-requirements.txt huggingface_hub

COPY --chown=user app/backend ./backend
COPY --chown=user app/frontend ./frontend
COPY --chown=user start.sh ./start.sh
RUN chmod +x start.sh

EXPOSE 7860
CMD ["./start.sh"]
