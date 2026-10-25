"""
Build the evaluation set from glaiveai/glaive-function-calling-v2.

Samples random rows (fixed seed) that offer at least one function and keeps the first
user turn plus the first assistant turn as the reference. Reads the dataset's ~100 MB
parquet export from the Hub (cached in eval/.cache/).

Two categories:
  call     - the reference answer is a <functioncall>
  no_call  - functions are available but the conversation never calls one
             (e.g. "I can't order a pizza, I can only translate text"). Rows whose first
             reply is plain text but which call a function in a later turn ("Sure, what
             length should the password be?") are skipped: calling right away there
             is a defensible answer, so they can't be scored automatically.

Usage:
    python eval/build_dataset.py --calls 60 --no-calls 40 --seed 1234
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq

DATASET = "glaiveai/glaive-function-calling-v2"
PARQUET_URL = (
    "https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
CACHE = Path(__file__).parent / ".cache" / "glaive-train.parquet"
OUT = Path(__file__).parent / "data" / "glaive_eval.jsonl"

ROLE_SPLIT = re.compile(r"(SYSTEM|USER|ASSISTANT|FUNCTION RESPONSE):\s*")
SINGLE_QUOTED_ARGS = re.compile(r"""("arguments"\s*:\s*)'(\{.*?\})'(\s*\})""", re.DOTALL)


def json_objects(text: str) -> list:
    decoder = json.JSONDecoder()
    found, i = [], 0
    while i < len(text):
        if text[i] == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
                found.append(obj)
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return found


def parse_row(index: int, row: dict) -> dict | None:
    system = re.sub(r"^SYSTEM:\s*", "", row["system"]).strip()
    if "with access to the following functions" not in system:
        return None
    functions = [f for f in json_objects(system) if "name" in f]
    if not functions:
        return None

    parts = ROLE_SPLIT.split(row["chat"])
    turns = [
        (parts[i], parts[i + 1].replace("<|endoftext|>", "").strip())
        for i in range(1, len(parts) - 1, 2)
    ]
    if len(turns) < 2 or turns[0][0] != "USER" or turns[1][0] != "ASSISTANT":
        return None
    user, assistant = turns[0][1], turns[1][1]

    expected_call = None
    if not assistant.startswith("<functioncall>") and "<functioncall>" in row["chat"]:
        return None  # deferred call: ambiguous, see module docstring
    if re.search(r"\bto=\w+:", row["chat"]):
        return None  # a few rows call functions in a stray "AI to=name: {...}" format
    if assistant.startswith("<functioncall>"):
        fixed = SINGLE_QUOTED_ARGS.sub(lambda m: m.group(1) + json.dumps(m.group(2)) + m.group(3), assistant)
        objs = json_objects(fixed)
        if not objs:
            return None
        call = objs[0]
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        expected_call = {"name": call["name"], "arguments": args or {}}

    return {
        "id": index,
        "category": "call" if expected_call else "no_call",
        "functions": functions,
        "user": user,
        "expected_call": expected_call,
        "reference_answer": assistant,
    }


def load_rows() -> list[dict]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {PARQUET_URL} ...")
        with httpx.stream("GET", PARQUET_URL, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with CACHE.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    return pq.read_table(CACHE).to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=60)
    ap.add_argument("--no-calls", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    want = {"call": args.calls, "no_call": args.no_calls}
    got: dict[str, list] = {"call": [], "no_call": []}
    rows = load_rows()
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)

    for index in order:
        if all(len(got[k]) >= want[k] for k in want):
            break
        sample = parse_row(index, rows[index])
        if sample and len(got[sample["category"]]) < want[sample["category"]]:
            got[sample["category"]].append(sample)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for sample in got["call"] + got["no_call"]:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}: {len(got['call'])} call, {len(got['no_call'])} no_call samples")


if __name__ == "__main__":
    main()
