#!/usr/bin/env bash
# Starts the web processes first (so the Space reports "Running" immediately),
# then downloads the GGUF and registers it with Ollama in the background.
set -euo pipefail

# Override either in Space settings → Variables. The repo has two Q4_K_M files whose
# names differ only in case; download just the one its Modelfile uses.
export GGUF_REPO="${GGUF_REPO:-AOUNZakaria/llama3-8b-function-calling-gguf}"
export GGUF_FILE="${GGUF_FILE:-llama-3-8b-Instruct.Q4_K_M.gguf}"

cd /home/user/app

ollama serve &

( cd backend && uvicorn main:app --host 127.0.0.1 --port 8000 ) &

(
  echo "Waiting for Ollama..."
  until curl -sf http://127.0.0.1:11434/api/tags >/dev/null; do sleep 1; done

  if ollama list | grep -q "^${OLLAMA_MODEL}:"; then
    echo "Model ${OLLAMA_MODEL} already registered."
  else
    echo "Downloading ${GGUF_FILE} from ${GGUF_REPO}..."
    GGUF_PATH=$(python - <<'PY'
import os
from huggingface_hub import hf_hub_download

print(hf_hub_download(
    repo_id=os.environ["GGUF_REPO"],
    filename=os.environ["GGUF_FILE"],
    local_dir="/home/user/gguf",
    token=os.environ.get("HF_TOKEN"),  # only needed for private repos
))
PY
    )
    echo "Using ${GGUF_PATH}"

    # TEMPLATE is just the prompt: backend/main.py already wraps messages in the
    # Llama 3 chat format (and sends raw=true), so Ollama must not apply a second one.
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
