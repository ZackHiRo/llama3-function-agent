#!/usr/bin/env bash
# Starts the web processes first (so the Space reports "Running" immediately),
# then downloads the GGUF and registers it with Ollama in the background.
set -euo pipefail

: "${GGUF_REPO:?Set the GGUF_REPO variable in Space settings, e.g. your-username/llama3-8b-function-calling-gguf}"

cd /home/user/app

ollama serve &

( cd backend && uvicorn main:app --host 127.0.0.1 --port 8000 ) &

(
  echo "Waiting for Ollama..."
  until curl -sf http://127.0.0.1:11434/api/tags >/dev/null; do sleep 1; done

  if ollama list | grep -q "^${OLLAMA_MODEL}"; then
    echo "Model ${OLLAMA_MODEL} already registered."
  else
    echo "Downloading GGUF from ${GGUF_REPO}..."
    GGUF_PATH=$(python - <<'PY'
import glob, os
from huggingface_hub import snapshot_download

folder = snapshot_download(
    repo_id=os.environ["GGUF_REPO"],
    allow_patterns=["*.gguf"],
    local_dir="/home/user/gguf",
    token=os.environ.get("HF_TOKEN"),  # only needed for private repos
)
print(sorted(glob.glob(f"{folder}/**/*.gguf", recursive=True))[0])
PY
    )
    echo "Using ${GGUF_PATH}"

    # TEMPLATE is just the prompt: backend/main.py already wraps messages in the
    # Llama 3 chat format, so Ollama must not apply a second template.
    cat > /home/user/Modelfile <<EOF
FROM ${GGUF_PATH}
TEMPLATE "{{ .Prompt }}"
PARAMETER stop "<|eot_id|>"
PARAMETER num_ctx 4096
PARAMETER temperature 0.7
EOF
    ollama create "${OLLAMA_MODEL}" -f /home/user/Modelfile
    echo "Model ${OLLAMA_MODEL} ready."
  fi
) &

# Streamlit runs in the foreground; the CORS/XSRF flags are needed behind the Space's iframe proxy.
exec streamlit run frontend/ui.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
