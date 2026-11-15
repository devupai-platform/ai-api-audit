# ai-api-audit

A 5-minute, vendor-neutral audit of any OpenAI-compatible AI API, run with your own key.

A pricing page can say anything. This script measures what the API actually does: which model answers, how your tokens are counted, whether limits hold, and whether streaming and long contexts really work.

Published by [DEVUP AI](https://devupai.com) under the MIT license. Run it against any provider, including DEVUP AI.

## What it checks

| # | Check | What a FAIL means |
|---|---|---|
| 1 | The model is listed by `GET /models` | You are paying for a name the API doesn't serve |
| 2 | The response names the model you asked for | Another model answered |
| 3 | Every response returns `usage`, and `total = prompt + completion` | Your bill can't be audited |
| 4 | The increase in `prompt_tokens` for a fixed reference paragraph matches the model's official tokenizer | Another model family is behind the name, or your prompt was rewritten |
| 5 | No large hidden text is added to your prompt (warning above 300 tokens) | You pay for tokens you never sent |
| 6 | `completion_tokens` matches the text you received | Output is overbilled |
| 7 | `max_tokens` is enforced | The spending limits in your code don't hold |
| 8 | Streaming works and its last chunk reports `usage` | Streaming apps can't track their cost |
| 9 | Opt-in: a passphrase placed at the start of a long prompt is recalled, and the whole prompt is counted | FAIL: fewer prompt tokens counted than sent, so the prompt was truncated. WARN: the whole prompt was counted but the model didn't recall the passphrase, a limit of the model's long-context recall |

When the response carries a per-request cost field (for example `_devup.cost_dzd` on DEVUP AI), the script also compares it with the published price in `/models` and the returned usage.

## How the tokenizer check works

Each model family has its own tokenizer, and the same text, especially Arabic or mixed-language text, splits into different token counts from one family to another. The script sends the same request twice, the second time with an extra reference paragraph, and compares the increase in `prompt_tokens` with the count from the official tokenizer. Using the difference cancels the chat template and any fixed system prompt, so the comparison is exact within a few tokens.

Limits: the check identifies the model family, not the numeric precision a model is served at. Closed models whose tokenizer isn't public can't be checked this way. Models that reason privately (reasoning tokens counted but not returned) skip check 6.

## Install and run

Python 3.8 or later. The script itself uses only the standard library.

```bash
pip install tokenizers          # open-weight models (downloads tokenizer.json from Hugging Face)
pip install tiktoken            # OpenAI models

export AI_API_KEY="your-api-key"    # PowerShell: $env:AI_API_KEY="your-api-key"

# Open-weight model
python ai_api_audit.py --base-url https://api.example.com/v1 \
  --model Qwen/Qwen2.5-72B-Instruct --hf-tokenizer Qwen/Qwen2.5-72B-Instruct

# OpenAI model
python ai_api_audit.py --base-url https://api.example.com/v1 \
  --model openai/gpt-4o-mini --tiktoken o200k_base

# Add the long-context check (sends about 100,000 input tokens: check the price first)
python ai_api_audit.py ... --context-test 100000 --yes

# Save the results
python ai_api_audit.py ... --json-out results.json
```

Without a tokenizer option, checks 4 to 6 report the raw numbers and skip the comparison.

Options: `--key-env` (default `AI_API_KEY`), `--max-overhead` (default 300), `--no-temperature` (for models that reject `temperature`), `--tokenizer-file` (a local `tokenizer.json`), `--timeout`.

## Cost and privacy

A default run sends five small requests, about 2,000 tokens in total, on the model you choose. The key is read from an environment variable and never printed or written to disk. The script contacts only the base URL you pass, plus Hugging Face or OpenAI's public tokenizer files when you ask for a tokenizer. No telemetry.

Exit code: `0` when no check fails, `1` when at least one check fails, `2` on a usage error.

## Tests

`test/run_tests.sh` starts a local mock provider in nine modes (honest, substituted model, missing usage, inflated output, hidden prompt text, truncated context, ignored `max_tokens`, wrong model name, weak long-context recall) and checks that each misbehaviour is caught and the honest mode passes. It needs `pip install tokenizers` and makes no network calls.
