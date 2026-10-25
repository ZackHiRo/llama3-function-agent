# Evaluation

How well does the QLoRA fine-tune turn requests into function calls, compared with the
base Llama 3 8B Instruct it started from?

## Results

<!-- results:start -->
_60 call samples + 40 out-of-scope samples_

| Metric | finetuned | base | base-instructed |
|---|---:|---:|---:|
| Call decision accuracy (all samples) | 99.0% | 40.0% | 97.0% |
| Function called when expected | 98.3% | 0.0% | 95.0% |
| Correct function name | 98.3% | 0.0% | 95.0% |
| Exact argument match | 88.3% | 0.0% | 85.0% |
| All expected arguments correct (extra optional args allowed) | 96.7% | 0.0% | 93.3% |
| Argument value accuracy | 96.7% | 0.0% | 94.4% |
| Glaive `<functioncall>` format | 98.3% | 0.0% | 93.3% |
| Correct abstention (out-of-scope requests) | 100.0% | 100.0% | 100.0% |
| Calls to non-existent functions | 0 | 0 | 0 |
| Median latency per sample (s) | 20.6 | 40.0 | 22.0 |
<!-- results:end -->

**Fully correct samples** (right call with all expected arguments, or a correct decline):
fine-tuned **98/100**, base with format instructions **96/100**, base without them **40/100**.

### Findings

1. **The fine-tune learned the calling protocol.** With the prompt the app sends, the
   fine-tuned model calls the right function on 59 of 60 requests and never calls a
   function on the 40 out-of-scope ones. Given that same prompt, the base model makes **no**
   calls at all. Instead it narrates the call and invents a result ("I'll use the
   `get_stock_price` function... According to the function, ..."). For an agent, that's
   the worst failure: the answer looks plausible and is made up.
2. **Prompt engineering closes most of that gap.** Once the system prompt spells out the
   call format and says not to make up results, the base model gets 96/100 fully correct
   versus 98/100 for the fine-tune. That 2-sample difference is **not statistically
   significant** (exact McNemar test on paired results, p = 0.69). The test set is
   in-distribution (Glaive-style), and a bigger or harder set would be needed to separate
   them.
3. **What the fine-tune still buys:** it needs no format instructions, so prompts are
   about 27% shorter (median 166 vs 228 tokens, which matters for CPU prefill). It also
   follows Glaive's `<functioncall>` format more consistently (98% vs 93%; the base model
   sometimes writes `generate_uuid {}` instead of JSON for no-argument calls).
4. **Most remaining errors are harmless.** Both models sometimes add optional arguments
   the reference left out (`"year": 2010` for *Inception*, `"dietary_restrictions": []`).
   That's why exact match (88%) is lower than "all expected arguments correct" (97%).
   The fine-tune's genuine failures were one call with broken nested quoting in
   `arguments`, and one case where it asked a clarifying question instead of calling
   `generate_username` with no arguments.

## Setup

| | |
|---|---|
| Fine-tuned model | [`AOUNZakaria/llama3-8b-function-calling-gguf`](https://huggingface.co/AOUNZakaria/llama3-8b-function-calling-gguf), `llama-3-8b-Instruct.Q4_K_M.gguf` |
| Baseline | Meta Llama 3 8B Instruct, same `Q4_K_M` quantization ([`bartowski/Meta-Llama-3-8B-Instruct-GGUF`](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF)) |
| Data | 100 samples from [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) (seed 1234), stored in [`data/glaive_eval.jsonl`](data/glaive_eval.jsonl) |
| Prompt and parser | The backend's own `format_llama3_prompt` and `extract_json_from_response` ([`app/backend/main.py`](../app/backend/main.py)) |
| Inference | llama-cpp-python 0.3.19 (llama.cpp, the engine inside Ollama), greedy decoding, CPU only |
| Hardware | MacBook Pro 2017, Intel i7-7820HQ (4 cores), 16 GB RAM |

Three configurations, all on the same 100 samples:

| Run | Model | Prompt |
|---|---|---|
| `finetuned` | QLoRA fine-tune | Glaive system prompt listing the functions as JSON schemas (what the app sends) |
| `base` | Base Llama 3 8B Instruct | Same prompt |
| `base-instructed` | Base Llama 3 8B Instruct | Same prompt, plus explicit instructions: the exact call format, "don't make up results", "answer in text if nothing fits" ([`run_eval.py`](run_eval.py) `FORMAT_INSTRUCTIONS`) |

`base-instructed` is there to answer the obvious question: is the fine-tune better than
just telling the base model what to do? The parser accepts a call in any reasonable JSON
shape (not only Glaive's `<functioncall>` prefix), so baselines aren't penalised for
formatting alone.

### The samples

- **60 call samples**: the reference reply is a function call. Scored on whether the model
  calls, picks the right function, and fills in the right arguments.
- **40 out-of-scope samples**: functions are available, but the request needs something else
  ("book me a flight" when the only function converts currencies). The reference reply
  declines in text, and the model should not call anything.

Rows whose first reply is text but which call a function a turn later (for example, the
assistant asks for a missing detail first) are left out, because calling right away is
also a reasonable answer there and can't be scored automatically.

### Metrics

| Metric | Meaning |
|---|---|
| Call decision accuracy | Called a function exactly when one was expected, over all 100 samples |
| Correct function name | Called the expected function (call samples) |
| Exact argument match | Right function and exactly the expected arguments; strings compared case-insensitively, `1000 == 1000.0` |
| Argument value accuracy | Share of expected argument values reproduced, averaged over call samples |
| Glaive `<functioncall>` format | Output follows the training format exactly |
| Correct abstention | No call on out-of-scope requests |
| Calls to non-existent functions | Calls to a function name that isn't in the prompt |

### Caveats

- **Small sample.** With 100 samples, a difference of a few points is within noise.
- **Overlap with training.** Glaive v2 has only a train split. The fine-tune ran 60 steps
  × 8 = 480 examples, about 0.4% of the 112,960 rows, so roughly 0–1 of these 100 samples
  is expected to have been seen during training.
- **Exact match is strict.** `"New York"` vs `"New York, NY"` counts as wrong, even though
  either would usually work.
- **Light fine-tune.** 60 steps is a short run. See the notebook for how to train longer.
- **One run per configuration.** Decoding is greedy, so reruns give the same outputs on
  the same build, but results may shift slightly on other llama.cpp versions or hardware.

## Reproduce

```bash
pip install llama-cpp-python fastapi pydantic httpx pyarrow  # scoring needs only the standard library

# 1. Build the eval set (already committed in data/)
python eval/build_dataset.py --calls 60 --no-calls 40 --seed 1234

# 2. Run each model (resumable; one JSON line per sample in eval/results/)
python eval/run_eval.py --model llama-3-8b-Instruct.Q4_K_M.gguf --label finetuned
python eval/run_eval.py --model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --label base
python eval/run_eval.py --model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --label base-instructed --instructed

# 3. Score and update the table above
python eval/score.py
```
