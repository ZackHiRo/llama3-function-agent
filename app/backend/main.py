"""
FastAPI Backend for Llama 3 Function Agent
Handles chat requests and interfaces with Ollama API
"""

import json
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Configuration
# ============================================================================

# Each value can be overridden via environment variable (used by the Hugging Face Space).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3-function-calling")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "120.0"))
NUM_CTX = int(os.environ.get("NUM_CTX", "4096"))

# The model was fine-tuned on glaiveai/glaive-function-calling-v2, whose system prompt
# lists the callable functions as JSON schemas. Prompting it the same way keeps it
# in-distribution; these are used when a request doesn't supply its own functions.
DEFAULT_FUNCTIONS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. Tokyo"},
                "units": {"type": "string", "enum": ["metric", "imperial"]},
            },
            "required": ["location"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "num_results": {"type": "integer", "description": "Number of results to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "schedule_meeting",
        "description": "Schedule a meeting in the calendar",
        "parameters": {
            "type": "object",
            "properties": {
                "participants": {"type": "array", "items": {"type": "string"}},
                "date": {"type": "string", "description": "Date of the meeting, e.g. tomorrow or 2024-06-01"},
                "time": {"type": "string", "description": "Start time, e.g. 10:00"},
                "duration_minutes": {"type": "integer"},
            },
            "required": ["participants", "date", "time"],
        },
    },
]

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
    functions: list[dict[str, Any]] | None = Field(
        default=None,
        description="JSON-schema function definitions the model may call (defaults to the built-in set)",
    )


class FunctionCall(BaseModel):
    """A function call extracted from the model output."""

    name: str = Field(..., description="Name of the function to call")
    arguments: Any = Field(default=None, description="Arguments for the call")


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
    function_call: FunctionCall | None = Field(
        default=None,
        description="Function call found in the output, if any",
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

    model_config = ConfigDict(protected_namespaces=())

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
    # Credentials can't be combined with a wildcard origin, and the API uses none.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================


def build_system_prompt(functions: list[dict[str, Any]]) -> str:
    """Build the system prompt in the Glaive format the model was fine-tuned on."""
    if not functions:
        return "You are a helpful assistant with no access to external functions."
    definitions = "\n\n".join(json.dumps(fn, indent=4) for fn in functions)  # Glaive separator
    return (
        "You are a helpful assistant with access to the following functions. "
        f"Use them if required -\n{definitions}"
    )


def format_llama3_prompt(user_message: str, functions: list[dict[str, Any]]) -> str:
    """
    Format a user message into the Llama 3 chat template.

    <|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\n\n

    No <|begin_of_text|>: llama.cpp (inside Ollama) adds BOS itself, and a manual one
    would give a double BOS. The notebook trains with a single BOS too.
    """
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{build_system_prompt(functions)}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def sanitize_input(message: str) -> str:
    """
    Sanitize user input to prevent prompt injection.

    Args:
        message: Raw user input

    Returns:
        Sanitized message string
    """
    # Remove Llama 3 control tokens (and any other <|...|> special token)
    sanitized = re.sub(r"<\|[a-z0-9_]+\|>", "", message)

    # Collapse runs of spaces/tabs but keep line breaks
    lines = (" ".join(line.split()) for line in sanitized.splitlines())
    return "\n".join(line for line in lines if line).strip()


# Glaive wraps arguments in single quotes, which isn't valid JSON:
#   <functioncall> {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}
_SINGLE_QUOTED_ARGS = re.compile(r"""("arguments"\s*:\s*)'(\{.*?\})'(\s*\})""", re.DOTALL)


def _find_json_values(text: str) -> list[Any]:
    """Return every top-level JSON object/array that can be decoded from text."""
    decoder = json.JSONDecoder()
    values = []
    i = 0
    while i < len(text):
        if text[i] in "{[":
            try:
                value, end = decoder.raw_decode(text, i)
                values.append(value)
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return values


def extract_function_call(parsed: Any) -> FunctionCall | None:
    """Pull a function call out of parsed JSON ({"name","arguments"} or {"action","parameters"})."""
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name") or parsed.get("action")
    if not name or not isinstance(name, str):
        return None
    args = parsed.get("arguments", parsed.get("parameters"))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return FunctionCall(name=name, arguments=args)


def extract_json_from_response(text: str) -> ParsedOutput:
    """
    Attempt to extract and parse JSON from model response.

    Args:
        text: Raw model response text

    Returns:
        ParsedOutput with parsed JSON or error details
    """
    normalized = _SINGLE_QUOTED_ARGS.sub(
        lambda m: m.group(1) + json.dumps(m.group(2)) + m.group(3), text
    )

    # Prefer fenced code blocks, then any JSON value in the text
    candidates: list[Any] = []
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", normalized):
        candidates.extend(_find_json_values(block))
    candidates.extend(_find_json_values(normalized))

    if not candidates:
        return ParsedOutput(
            raw_text=text,
            parsed_json=None,
            parse_error="No valid JSON found in the response",
        )

    # A function call beats any other JSON the model might have echoed
    parsed = next((c for c in candidates if extract_function_call(c)), candidates[0])
    return ParsedOutput(
        raw_text=text,
        parsed_json=parsed,
        parse_error=None,
        function_call=extract_function_call(parsed),
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
        # The prompt is already in the chat template; raw mode stops Ollama from
        # wrapping it in the Modelfile's TEMPLATE/SYSTEM a second time.
        "raw": True,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": NUM_CTX,
            "stop": ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],
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
            # Ollama's own status (e.g. 404 "model not found") would read as this API's
            # status, so report it as a bad gateway with Ollama's message.
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ollama API error ({e.response.status_code}): {e.response.text}",
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Error talking to Ollama: {e}",
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
            names = [m.get("name", "") for m in data.get("models", [])]
            # Ollama reports "name:tag"; accept OLLAMA_MODEL with or without a tag
            model_available = any(
                OLLAMA_MODEL in (name, name.split(":")[0]) for name in names
            )

            return True, model_available

        except (httpx.RequestError, ValueError):
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
        502: {"model": ErrorResponse, "description": "Ollama returned an error"},
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
    2. Formats it into the Llama 3 chat template with the function definitions
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
    functions = DEFAULT_FUNCTIONS if request.functions is None else request.functions
    formatted_prompt = format_llama3_prompt(sanitized_message, functions)

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


@app.get("/functions", tags=["Chat"], summary="Default callable functions")
async def list_functions() -> list[dict[str, Any]]:
    """Function definitions used when a chat request doesn't supply its own."""
    return DEFAULT_FUNCTIONS


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "Llama 3 Function Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }
