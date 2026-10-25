"""
Run a GGUF model over the eval set and record one JSON line per sample.

Prompts are built and outputs parsed with the same functions the FastAPI backend uses
(app/backend/main.py), so this measures the production prompt + parser, not a
separate re-implementation. Inference runs on llama-cpp-python (the llama.cpp engine
Ollama wraps) with greedy decoding so runs are reproducible.

Results are appended to eval/results/<label>.jsonl; re-running skips samples that are
already there, so an interrupted run can be resumed.

Usage:
    python eval/run_eval.py --model path/to/model.gguf --label finetuned
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from llama_cpp import Llama

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "app" / "backend"))
from main import (  # noqa: E402
    build_system_prompt,
    extract_json_from_response,
    format_llama3_prompt,
    sanitize_input,
)

DATA = ROOT / "data" / "glaive_eval.jsonl"
RESULTS = ROOT / "results"
STOP = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]

# --instructed: spell out the call format for models that weren't trained on it, so the
# baseline is "base model + prompt engineering" rather than "base model left guessing".
FORMAT_INSTRUCTIONS = (
    "\n\nTo call a function, reply with only this line and nothing else:\n"
    '<functioncall> {"name": "<function name>", "arguments": {<arguments as JSON>}}\n'
    "Do not make up function results. If none of the functions can help with the request, "
    "reply in plain text without calling a function."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a .gguf file")
    ap.add_argument("--label", required=True, help="Name for this run, e.g. finetuned or base")
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N samples")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--instructed", action="store_true", help="Add explicit call-format instructions")
    args = ap.parse_args()

    samples = [json.loads(line) for line in DATA.open()]
    if args.limit:
        samples = samples[: args.limit]

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{args.label}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.open()}

    llm = Llama(
        model_path=args.model,
        n_ctx=4096,
        n_threads=args.threads,
        n_gpu_layers=0,
        seed=0,
        verbose=False,
    )

    with out_path.open("a") as out:
        for i, sample in enumerate(samples, 1):
            if sample["id"] in done:
                continue
            prompt = format_llama3_prompt(sanitize_input(sample["user"]), sample["functions"])
            if args.instructed:
                system = build_system_prompt(sample["functions"])
                prompt = prompt.replace(system, system + FORMAT_INSTRUCTIONS, 1)
            started = time.perf_counter()
            result = llm.create_completion(
                prompt,
                max_tokens=args.max_tokens,
                temperature=0.0,
                stop=STOP,
            )
            latency = time.perf_counter() - started

            text = result["choices"][0]["text"]
            parsed = extract_json_from_response(text)
            call = parsed.function_call.model_dump() if parsed.function_call else None
            record = {
                "id": sample["id"],
                "category": sample["category"],
                "output": text,
                "predicted_call": call,
                "expected_call": sample["expected_call"],
                "function_names": [f["name"] for f in sample["functions"]],
                "latency_s": round(latency, 2),
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            shown = call["name"] if call else "(no call)"
            print(f"[{args.label}] {i}/{len(samples)} {sample['category']:8} -> {shown:28} {latency:5.1f}s", flush=True)


if __name__ == "__main__":
    main()
