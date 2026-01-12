"""
FastAPI Backend for Llama 3 Function Agent
Handles chat requests and interfaces with Ollama API
"""

import json
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================================
# Configuration
# ============================================================================

OLLAMA_BASE_URL = "http://host.docker.internal:11434"
OLLAMA_MODEL = "llama3"
REQUEST_TIMEOUT = 120.0

# ============================================================================
# Pydantic Models
# ============================================================================


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="User message to send to the model",
        examples=["Get weather in Tokyo"],
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for response generation",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Maximum tokens in the response",
    )


class ParsedOutput(BaseModel):
    """Parsed JSON output from model response."""

    raw_text: str = Field(..., description="Raw text response from the model")
    parsed_json: dict[str, Any] | list[Any] | None = Field(
        default=None,
        description="Parsed JSON if model returned valid JSON",
    )
    parse_error: str | None = Field(
        default=None,
        description="Error message if JSON parsing failed",
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    success: bool = Field(..., description="Whether the request was successful")
    response: str = Field(..., description="Model response text")
    parsed_output: ParsedOutput = Field(..., description="Parsed output details")
    model: str = Field(..., description="Model used for generation")
    tokens_used: int | None = Field(
        default=None,
        description="Number of tokens used in response",
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="API health status")
    ollama_connected: bool = Field(..., description="Ollama connection status")
    model_available: bool = Field(..., description="Whether the model is loaded")


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description="Error details")


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Llama 3 Function Agent API",
    description="API for interacting with fine-tuned Llama 3 model via Ollama",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================


def format_llama3_prompt(user_message: str) -> str:
    """
    Format user message into Llama 3 ChatML template.

    The Llama 3 chat format uses special tokens:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_message}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {user_message}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    system_prompt = """You are a helpful AI assistant that can perform function calls.
When asked to perform actions, respond with a JSON object containing:
- "action": the action to perform
- "parameters": an object with relevant parameters
- "reasoning": brief explanation of your approach

Always respond with valid JSON when performing function calls."""

    formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|}

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return formatted


def sanitize_input(message: str) -> str:
    """
    Sanitize user input to prevent prompt injection.

    Args:
        message: Raw user input

    Returns:
        Sanitized message string
    """
    # Remove potentially dangerous special tokens
    dangerous_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|finetune_right_pad_id|>",
    ]

    sanitized = message
    for token in dangerous_tokens:
        sanitized = sanitized.replace(token, "")

    # Remove excessive whitespace
    sanitized = " ".join(sanitized.split())

    return sanitized.strip()


def extract_json_from_response(text: str) -> ParsedOutput:
    """
    Attempt to extract and parse JSON from model response.

    Args:
        text: Raw model response text

    Returns:
        ParsedOutput with parsed JSON or error details
    """
    # Try to find JSON in code blocks first
    json_patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"\{[\s\S]*\}",
        r"\[[\s\S]*\]",
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                parsed = json.loads(match)
                return ParsedOutput(
                    raw_text=text,
                    parsed_json=parsed,
                    parse_error=None,
                )
            except json.JSONDecodeError:
                continue

    # If no valid JSON found, try parsing the entire response
    try:
        parsed = json.loads(text.strip())
        return ParsedOutput(
            raw_text=text,
            parsed_json=parsed,
            parse_error=None,
        )
    except json.JSONDecodeError as e:
        return ParsedOutput(
            raw_text=text,
            parsed_json=None,
            parse_error=f"Could not parse JSON: {str(e)}",
        )


async def call_ollama_api(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> tuple[str, int | None]:
    """
    Call the Ollama API with the formatted prompt.

    Args:
        prompt: Formatted prompt string
        temperature: Sampling temperature
        max_tokens: Maximum response tokens

    Returns:
        Tuple of (response_text, tokens_used)

    Raises:
        HTTPException: If Ollama API call fails
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return (
                data.get("response", ""),
                data.get("eval_count"),
            )

        except httpx.ConnectError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Cannot connect to Ollama at {OLLAMA_BASE_URL}: {str(e)}",
            )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Ollama request timed out",
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ollama API error: {e.response.text}",
            )


async def check_ollama_connection() -> tuple[bool, bool]:
    """
    Check if Ollama is reachable and model is available.

    Returns:
        Tuple of (ollama_connected, model_available)
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            # Check if Ollama is running
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                return False, False

            # Check if our model is available
            data = response.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            model_available = OLLAMA_MODEL in models

            return True, model_available

        except (httpx.ConnectError, httpx.TimeoutException):
            return False, False


# ============================================================================
# API Endpoints
# ============================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
)
async def health_check() -> HealthResponse:
    """
    Check API health and Ollama connection status.
    """
    ollama_connected, model_available = await check_ollama_connection()

    return HealthResponse(
        status="healthy",
        ollama_connected=ollama_connected,
        model_available=model_available,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Ollama unavailable"},
        504: {"model": ErrorResponse, "description": "Request timeout"},
    },
    tags=["Chat"],
    summary="Send a message to Llama 3",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a user message to the Llama 3 model and receive a response.

    The endpoint:
    1. Sanitizes the user input
    2. Formats it into the Llama 3 ChatML template
    3. Sends it to Ollama for inference
    4. Attempts to parse JSON from the response
    5. Returns structured output
    """
    # Sanitize input
    sanitized_message = sanitize_input(request.message)

    if not sanitized_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message is empty after sanitization",
        )

    # Format prompt
    formatted_prompt = format_llama3_prompt(sanitized_message)

    # Call Ollama
    response_text, tokens_used = await call_ollama_api(
        prompt=formatted_prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    # Parse response
    parsed_output = extract_json_from_response(response_text)

    return ChatResponse(
        success=True,
        response=response_text,
        parsed_output=parsed_output,
        model=OLLAMA_MODEL,
        tokens_used=tokens_used,
    )


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "Llama 3 Function Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }
