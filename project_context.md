# Project: Llama 3 Function Calling Agent

## Goal
Build a production-ready inference server and UI for a Llama 3 8B model fine-tuned on the "Glaive Function Calling" dataset.

## Architecture
1. **Model:** Llama 3 8B (Quantized to GGUF format). Trained on Google Colab, exported to this server.
2. **Inference Engine:** Ollama (running locally on DigitalOcean Linux).
3. **Backend:** FastAPI (Handling validation and routing requests to Ollama).
4. **Frontend:** Streamlit (User Interface).
5. **Deployment:** DigitalOcean Droplet (Ubuntu).

## Data Format (ChatML)
The model expects this specific prompt format:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant designed to output JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

## Expected Output
The model will output raw JSON. The UI must parse this JSON and display it prettified.