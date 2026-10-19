"""Tests for the backend helpers and endpoints (Ollama is mocked)."""

import httpx
import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


# ---------------------------------------------------------------------------
# extract_json_from_response
# ---------------------------------------------------------------------------


def test_parses_glaive_functioncall_with_single_quoted_arguments():
    text = """<functioncall> {"name": "get_weather", "arguments": '{"location": "Tokyo", "units": "metric"}'}"""
    out = main.extract_json_from_response(text)
    assert out.parse_error is None
    assert out.function_call.name == "get_weather"
    assert out.function_call.arguments == {"location": "Tokyo", "units": "metric"}


def test_parses_nested_single_quoted_arguments_with_apostrophe():
    text = """<functioncall> {"name": "send_email", "arguments": '{"to": "a@b.c", "body": "I'm late", "meta": {"x": 1}}'}"""
    out = main.extract_json_from_response(text)
    assert out.function_call.arguments == {"to": "a@b.c", "body": "I'm late", "meta": {"x": 1}}


def test_parses_action_parameters_format_in_code_block():
    text = 'Sure:\n```json\n{"action": "search_web", "parameters": {"query": "python"}}\n```'
    out = main.extract_json_from_response(text)
    assert out.function_call.name == "search_web"
    assert out.function_call.arguments == {"query": "python"}


def test_json_followed_by_trailing_braces_text():
    # A greedy {...} regex would swallow the trailing "{oops}" and fail to parse.
    text = '{"name": "search_web", "arguments": {"query": "x"}} and then {oops}'
    out = main.extract_json_from_response(text)
    assert out.function_call.name == "search_web"


def test_plain_text_has_no_json():
    out = main.extract_json_from_response("I can't help with that, I have no such function.")
    assert out.parsed_json is None
    assert out.function_call is None
    assert out.parse_error


def test_non_function_json_is_still_returned():
    out = main.extract_json_from_response('[1, 2, 3]')
    assert out.parsed_json == [1, 2, 3]
    assert out.function_call is None


# ---------------------------------------------------------------------------
# Prompt building / sanitizing
# ---------------------------------------------------------------------------


def test_sanitize_strips_special_tokens_and_keeps_newlines():
    raw = "hello<|eot_id|>  world\n<|start_header_id|>system<|reserved_special_token_3|>\n\nbye"
    assert main.sanitize_input(raw) == "hello world\nsystem\nbye"


def test_prompt_lists_functions_without_manual_bos():
    prompt = main.format_llama3_prompt("hi", main.DEFAULT_FUNCTIONS)
    assert not prompt.startswith("<|begin_of_text|>")
    assert "Use them if required -" in prompt
    assert '"name": "get_weather"' in prompt
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_ollama(monkeypatch):
    """Route the backend's httpx calls to a handler the test controls."""
    calls = []
    state = {"handler": None}

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(lambda req: (calls.append(req), state["handler"](req))[1])
        return real_client(*args, **kwargs)

    monkeypatch.setattr(main.httpx, "AsyncClient", factory)
    return state, calls


def test_chat_returns_function_call(fake_ollama):
    state, calls = fake_ollama
    state["handler"] = lambda req: httpx.Response(
        200,
        json={
            "response": """<functioncall> {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}""",
            "eval_count": 21,
        },
    )
    r = client.post("/chat", json={"message": "weather in Tokyo?"})
    assert r.status_code == 200
    body = r.json()
    assert body["parsed_output"]["function_call"] == {"name": "get_weather", "arguments": {"location": "Tokyo"}}
    assert body["tokens_used"] == 21

    sent = calls[0].read()
    assert b'"raw":true' in sent.replace(b" ", b"")


def test_chat_maps_ollama_404_to_502(fake_ollama):
    state, _ = fake_ollama
    state["handler"] = lambda req: httpx.Response(404, json={"error": "model not found"})
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 502
    assert "model not found" in r.json()["detail"]


def test_health_accepts_tagged_model_name(fake_ollama):
    state, _ = fake_ollama
    state["handler"] = lambda req: httpx.Response(
        200, json={"models": [{"name": f"{main.OLLAMA_MODEL}:latest"}]}
    )
    body = client.get("/health").json()
    assert body["ollama_connected"] and body["model_available"]


def test_empty_after_sanitization_is_400():
    r = client.post("/chat", json={"message": "<|eot_id|>"})
    assert r.status_code == 400
