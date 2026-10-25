"""
Score eval/results/*.jsonl and write the results table into eval/README.md.

Metrics
  call samples (a function call is expected):
    called            - the model emitted a parsable function call
    name accuracy     - called the expected function
    args exact match  - called the expected function with exactly the expected arguments
                        (strings compared case-insensitively, 1000 == 1000.0)
    args superset     - every expected argument is right; extra optional arguments allowed
    arg value accuracy- share of expected argument values reproduced, averaged per sample
    Glaive format     - output starts with "<functioncall>" like the training data
  no_call samples (the request is out of scope for the given functions):
    correct abstention- the model answered in text without calling a function
  overall:
    call decision accuracy - called iff a call was expected, over all samples
    hallucinated names     - calls to a function that isn't in the prompt

Usage:
    python eval/score.py            # scores every results file
"""

import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "glaive_eval.jsonl"
REPORT = ROOT / "README.md"


def norm(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError:
            return stripped.lower()
    if isinstance(value, list):
        return [norm(v) for v in value]
    if isinstance(value, dict):
        return {k: norm(v) for k, v in value.items()}
    return value


def score(records: list[dict]) -> dict:
    calls = [r for r in records if r["category"] == "call"]
    no_calls = [r for r in records if r["category"] == "no_call"]

    def called(r):
        return r["predicted_call"] is not None

    def name_ok(r):
        return called(r) and r["predicted_call"]["name"] == r["expected_call"]["name"]

    def args_of(r):
        args = r["predicted_call"].get("arguments")
        return args if isinstance(args, dict) else {}

    def args_exact(r):
        return name_ok(r) and norm(args_of(r)) == norm(r["expected_call"]["arguments"])

    def arg_value_acc(r):
        expected = r["expected_call"]["arguments"] or {}
        if not expected:
            return 1.0 if name_ok(r) else 0.0
        if not name_ok(r):
            return 0.0
        got = args_of(r)
        return sum(norm(got.get(k)) == norm(v) for k, v in expected.items()) / len(expected)

    def args_superset(r):
        expected = norm(r["expected_call"]["arguments"] or {})
        got = norm(args_of(r)) if name_ok(r) else None
        return got is not None and all(k in got and got[k] == v for k, v in expected.items())

    def hallucinated(r):
        return called(r) and r["predicted_call"]["name"] not in r["function_names"]

    def pct(xs):
        return 100 * mean(xs) if xs else float("nan")

    decisions = [called(r) == (r["category"] == "call") for r in records]
    return {
        "n_call": len(calls),
        "n_no_call": len(no_calls),
        "called": pct([called(r) for r in calls]),
        "name_acc": pct([name_ok(r) for r in calls]),
        "args_exact": pct([args_exact(r) for r in calls]),
        "args_superset": pct([args_superset(r) for r in calls]),
        "arg_value_acc": pct([arg_value_acc(r) for r in calls]),
        "glaive_format": pct([r["output"].lstrip().startswith("<functioncall>") for r in calls]),
        "abstain": pct([not called(r) for r in no_calls]),
        "decision_acc": pct(decisions),
        "hallucinated": sum(hallucinated(r) for r in records),
        "median_latency": median(r["latency_s"] for r in records),
    }


ROWS = [
    ("Call decision accuracy (all samples)", "decision_acc", "%"),
    ("Function called when expected", "called", "%"),
    ("Correct function name", "name_acc", "%"),
    ("Exact argument match", "args_exact", "%"),
    ("All expected arguments correct (extra optional args allowed)", "args_superset", "%"),
    ("Argument value accuracy", "arg_value_acc", "%"),
    ("Glaive `<functioncall>` format", "glaive_format", "%"),
    ("Correct abstention (out-of-scope requests)", "abstain", "%"),
    ("Calls to non-existent functions", "hallucinated", "n"),
    ("Median latency per sample (s)", "median_latency", "s"),
]


def main() -> None:
    # Only score samples that are in the current eval set
    ids = {json.loads(line)["id"] for line in DATA.open()}
    runs = {
        p.stem: [r for r in map(json.loads, p.open()) if r["id"] in ids]
        for p in sorted(RESULTS.glob("*.jsonl"))
    }
    if not runs:
        raise SystemExit("No results in eval/results/. Run eval/run_eval.py first.")
    # Show the fine-tuned model first if present
    labels = sorted(runs, key=lambda k: (k != "finetuned", k))
    scores = {label: score(runs[label]) for label in labels}

    first = scores[labels[0]]
    lines = [
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for title, key, kind in ROWS:
        cells = []
        for label in labels:
            v = scores[label][key]
            cells.append(f"{v:.1f}%" if kind == "%" else (str(v) if kind == "n" else f"{v:.1f}"))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    counts = f"{first['n_call']} call samples + {first['n_no_call']} out-of-scope samples"
    print(f"{counts}\n\n{table}")

    marker_start, marker_end = "<!-- results:start -->", "<!-- results:end -->"
    block = f"{marker_start}\n_{counts}_\n\n{table}\n{marker_end}"
    if REPORT.exists() and marker_start in REPORT.read_text():
        text = REPORT.read_text()
        head, rest = text.split(marker_start, 1)
        tail = rest.split(marker_end, 1)[1]
        REPORT.write_text(head + block + tail)
    else:
        REPORT.write_text(f"# Evaluation results\n\n{block}\n")
    print(f"\nUpdated {REPORT}")


if __name__ == "__main__":
    main()
