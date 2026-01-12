# 🦙 Llama 3 Function Agent

A production-ready full-stack AI application for deploying a fine-tuned Llama 3 model on DigitalOcean. Built with FastAPI, Streamlit, and Ollama.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [DigitalOcean Deployment](#-digitalocean-deployment)
- [Local Development](#-local-development)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

## 🎯 Overview

This application provides a clean interface for interacting with Llama 3 for function calling capabilities. Users can input natural language commands and receive structured JSON responses.

### Features

- **FastAPI Backend**: High-performance async API with automatic OpenAPI documentation
- **Streamlit Frontend**: Clean two-column UI with real-time status indicators
- **Ollama Integration**: Seamless connection to locally running Llama 3 models
- **JSON Parsing**: Automatic extraction and formatting of JSON from model responses
- **Input Sanitization**: Protection against prompt injection attacks
- **Docker Compose**: One-command deployment with health checks

## 🏗 Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   Streamlit     │────▶│    FastAPI      │────▶│    Ollama       │
│   Frontend      │     │    Backend      │     │    (Llama 3)    │
│   Port: 8501    │     │    Port: 8000   │     │    Port: 11434  │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 📦 Prerequisites

- **Docker** & **Docker Compose** (v2.0+)
- **Ollama** installed on host or running as container
- **Llama 3** model pulled in Ollama
- **4GB+ RAM** (8GB+ recommended for larger models)

## 🌊 DigitalOcean Deployment

### Step 1: Create a Droplet

1. Log in to [DigitalOcean](https://cloud.digitalocean.com/)
2. Create a new Droplet:
   - **Image**: Ubuntu 22.04 LTS
   - **Size**: CPU-Optimized (c-4 or higher recommended)
   - **Region**: Choose closest to your users
   - **Authentication**: SSH key (recommended)

> 💡 For GPU inference, consider DigitalOcean's GPU Droplets or use a CPU-optimized instance with quantized models.

### Step 2: Install Docker

SSH into your Droplet and run:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose plugin
sudo apt install docker-compose-plugin -y

# Verify installation
docker --version
docker compose version

# Log out and back in for group changes
exit
```

### Step 3: Install Ollama

```bash
# SSH back into the Droplet
ssh root@your-droplet-ip

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify Ollama is running
ollama --version

# Check Ollama service status
sudo systemctl status ollama
```

### Step 4: Pull the Llama 3 Model

```bash
# Pull the base Llama 3 model (8B parameters)
ollama pull llama3

# For the 70B model (requires more RAM):
# ollama pull llama3:70b

# For custom GGUF models:
# 1. Create a Modelfile
cat << 'EOF' > Modelfile
FROM /path/to/your-model.gguf

PARAMETER temperature 0.7
PARAMETER num_ctx 4096

TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
EOF

# 2. Create the model
ollama create my-llama3 -f Modelfile

# Verify model is available
ollama list
```

### Step 5: Deploy the Application

```bash
# Clone the repository
git clone https://github.com/yourusername/llama3-function-agent.git
cd llama3-function-agent

# Build and start services
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps
```

### Step 6: Configure Firewall

```bash
# Allow HTTP traffic
sudo ufw allow 8501/tcp  # Streamlit
sudo ufw allow 8000/tcp  # FastAPI (optional, for API access)
sudo ufw enable
```

### Step 7: Access the Application

- **Frontend**: `http://your-droplet-ip:8501`
- **API Docs**: `http://your-droplet-ip:8000/docs`

## 💻 Local Development

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/llama3-function-agent.git
cd llama3-function-agent

# Ensure Ollama is running with Llama 3
ollama pull llama3
ollama serve  # If not running as service

# Start the application
docker compose up --build
```

### Development Mode (Without Docker)

```bash
# Backend
cd app/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd app/frontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run ui.py
```

## 📚 API Reference

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "ollama_connected": true,
  "model_available": true
}
```

### Chat Endpoint

```http
POST /chat
Content-Type: application/json

{
  "message": "Get weather in Tokyo",
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Response:**
```json
{
  "success": true,
  "response": "```json\n{\"action\": \"get_weather\", ...}\n```",
  "parsed_output": {
    "raw_text": "...",
    "parsed_json": {
      "action": "get_weather",
      "parameters": {
        "location": "Tokyo",
        "units": "metric"
      },
      "reasoning": "User requested weather information for Tokyo"
    },
    "parse_error": null
  },
  "model": "llama3",
  "tokens_used": 156
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3` | Model name in Ollama |
| `REQUEST_TIMEOUT` | `120.0` | Request timeout in seconds |

### Modifying the Backend

Edit `app/backend/main.py` to change:

```python
OLLAMA_BASE_URL = "http://host.docker.internal:11434"  # Ollama endpoint
OLLAMA_MODEL = "llama3"  # Model name
REQUEST_TIMEOUT = 120.0  # Timeout in seconds
```

### Using a Custom Model

1. Create your model in Ollama:
   ```bash
   ollama create my-custom-llama3 -f Modelfile
   ```

2. Update the backend configuration:
   ```python
   OLLAMA_MODEL = "my-custom-llama3"
   ```

3. Rebuild the containers:
   ```bash
   docker compose up --build -d
   ```

## 🔧 Troubleshooting

### Ollama Connection Issues

**Symptom**: "Cannot connect to Ollama" error

**Solutions**:

1. Verify Ollama is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. On Linux, ensure `host.docker.internal` is configured:
   ```yaml
   # docker-compose.yml
   extra_hosts:
     - "host.docker.internal:host-gateway"
   ```

3. Check firewall rules:
   ```bash
   sudo ufw allow 11434/tcp
   ```

### Model Not Loading

**Symptom**: "Model Not Loaded" in status panel

**Solutions**:

1. Verify model is pulled:
   ```bash
   ollama list
   ```

2. Pull the model if missing:
   ```bash
   ollama pull llama3
   ```

3. Check available memory:
   ```bash
   free -h
   ```

### Docker Build Failures

**Solutions**:

1. Clear Docker cache:
   ```bash
   docker compose build --no-cache
   ```

2. Prune unused resources:
   ```bash
   docker system prune -a
   ```

### Slow Response Times

**Solutions**:

1. Use a quantized model (e.g., `llama3:8b-q4_0`)
2. Increase Droplet resources
3. Adjust `max_tokens` in requests

## 📁 Project Structure

```
llama3-function-agent/
├── app/
│   ├── backend/
│   │   ├── main.py           # FastAPI application
│   │   ├── requirements.txt  # Python dependencies
│   │   └── Dockerfile        # Backend container
│   └── frontend/
│       ├── ui.py             # Streamlit application
│       ├── requirements.txt  # Python dependencies
│       └── Dockerfile        # Frontend container
├── notebooks/
│   └── finetune.ipynb        # Llama 3 fine-tuning notebook
├── docker-compose.yml        # Service orchestration
├── README.md                 # Documentation
└── project_context.md        # Project context
```

## 🎓 Fine-Tuning Guide

The `notebooks/finetune.ipynb` notebook provides a complete tutorial for fine-tuning Llama 3 8B for function calling:

### Features

- **QLoRA Training**: Memory-efficient 4-bit quantization with LoRA adapters
- **Unsloth Optimization**: 2x faster training, 50% less memory
- **Google Colab Compatible**: Works on free T4 GPU tier
- **GGUF Export**: Ready for Ollama deployment

### Quick Start (Colab)

1. Open `notebooks/finetune.ipynb` in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially
4. Download the GGUF file for Ollama deployment

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `unsloth/llama-3-8b-Instruct-bnb-4bit` |
| Dataset | `glaiveai/glaive-function-calling-v2` |
| Learning Rate | 2e-4 |
| Batch Size | 2 (effective: 8 with gradient accumulation) |
| LoRA Rank | 16 |
| Training Steps | 60 (increase for production) |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<p align="center">
  Built with ❤️ using FastAPI, Streamlit, and Ollama
</p>
