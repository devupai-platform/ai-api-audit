#!/usr/bin/env python3
"""
ai_api_audit.py: a 5-minute, vendor-neutral audit of any OpenAI-compatible AI API.

It checks, on your own key and your own account, what a provider's marketing page cannot prove:
  1. the model you asked for is listed by the API itself;
  2. the response names the model you asked for;
  3. every response carries usage (prompt / completion / total tokens) and the numbers add up;
  4. the provider counts YOUR prompt with the claimed model's own tokenizer
     (a different tokenizer means a different model family behind the name);
  5. no large hidden text is added to your prompt that you pay for;
  6. completion tokens match the text you actually received (no inflated output billing);
  7. max_tokens is enforced;
  8. streaming works and reports usage;
  9. optional: the advertised context window is really served (needle test, opt-in, costs more tokens):
     FAIL when fewer prompt tokens are counted than sent (truncation), WARN when the full prompt is
     counted but the model does not recall the passphrase (a limit of the model's long-context recall).

Default run: 5 small requests, about 2,000 tokens in total on the chosen model.
The key is read from the environment and is never printed.

Usage:
  export AI_API_KEY="..."
  python ai_api_audit.py --base-url https://api.example.com/v1 --model org/model-name \
      --hf-tokenizer org/model-name            # open-weight models (pip install tokenizers)
  python ai_api_audit.py ... --tiktoken o200k_base   # OpenAI models (pip install tiktoken)

Exit code: 0 when no check FAILs, 1 when at least one check FAILs, 2 on a usage error.
License: MIT. No telemetry. Only the base URL you pass is contacted
(plus Hugging Face or OpenAI's public tokenizer files when you ask for a tokenizer).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

VERSION = "1.1.0"
UA = f"ai-api-audit/{VERSION}"

# Mixed Arabic / French / English / code / numbers. Different model families split this text
# into very different numbers of tokens, which is what makes the tokenizer check discriminating.
FILLER = (
    "ملاحظات مرجعية: تعتمد المؤسسات الجزائرية بشكل متزايد على واجهات الذكاء الاصطناعي لأتمتة "
    "خدمة الزبائن، وتحليل الوثائق الإدارية، وكتابة البرمجيات. يجب أن تكون الفوترة شفافة، "
    "وأن يُحتسب كل رمز بدقة، وأن تُحفظ البيانات وفق القانون 18-07.\n"
    "Notes de référence : une passerelle d'API sérieuse publie ses tarifs, ses limites de débit, "
    "son historique de disponibilité et ses conditions d'utilisation. Elle renvoie l'usage exact "
    "de chaque requête, y compris les jetons mis en cache.\n"
    "Reference notes: a production gateway returns usage on every response, enforces max_tokens, "
    "streams server-sent events, and lists the real context window of each model it serves.\n"
    "def total_cost(prompt_tokens: int, completion_tokens: int, p_in: float, p_out: float) -> float:\n"
    "    return prompt_tokens / 1_000_000 * p_in + completion_tokens / 1_000_000 * p_out\n"
    "Inventory 2026-11-15: 418 items, 57 classes, 63 guides; ratios 0.41, 0.58, 0.76; "
    "IDs A7-4410, B3-0092, C9-7781.\n"
)
BASE_PROMPT = "Write exactly three short sentences about the Sahara desert. Plain text, no list."
SEP = "\n\n"
NEEDLE = "The audit passphrase is CEDAR-7341-ALGIERS."

PASS, WARN, FAIL, INFO, SKIP = "PASS", "WARN", "FAIL", "INFO", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)


class ApiError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body


# ----------------------------------------------------------------------------- HTTP

class Client:
    def __init__(self, base_url: str, key: str, timeout: float):
        self.base = base_url.rstrip("/")
        self.key = key
        self.timeout = timeout

    def _req(self, method: str, path: str, body: Optional[dict] = None) -> urllib.request.Request:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.key}")
        req.add_header("User-Agent", UA)
        req.add_header("Accept", "application/json, text/event-stream")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        return req

    def get_json(self, path: str) -> Tuple[dict, float]:
        return self._json(self._req("GET", path))

    def post_json(self, path: str, body: dict) -> Tuple[dict, float]:
        return self._json(self._req("POST", path, body))

    def _json(self, req: urllib.request.Request) -> Tuple[dict, float]:
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, _safe_body(e.read()))
        dt = time.monotonic() - t0
        try:
            return json.loads(raw.decode("utf-8")), dt
        except ValueError:
            raise ApiError(200, "response is not JSON: " + _safe_body(raw))

    def post_stream(self, path: str, body: dict) -> Tuple[List[dict], float, float, bool]:
        """Returns (events, time_to_first_content_s, total_s, saw_done)."""
        req = self._req("POST", path, body)
        t0 = time.monotonic()
        ttfc: Optional[float] = None
        events: List[dict] = []
        saw_done = False
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                for raw_line in r:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        saw_done = True
                        break
                    try:
                        ev = json.loads(payload)
                    except ValueError:
                        continue
                    events.append(ev)
                    if ttfc is None and _delta_text(ev):
                        ttfc = time.monotonic() - t0
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, _safe_body(e.read()))
        total = time.monotonic() - t0
        return events, (ttfc if ttfc is not None else total), total, saw_done


def _safe_body(raw: bytes, limit: int = 400) -> str:
    s = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    s = " ".join(s.split())
    return s[:limit] + ("…" if len(s) > limit else "")


def _delta_text(ev: dict) -> str:
    out = ""
    for ch in ev.get("choices") or []:
        d = ch.get("delta") or {}
        out += (d.get("content") or "") + (d.get("reasoning_content") or "")
    return out


# ----------------------------------------------------------------------------- tokenizers

def load_tokenizer(args) -> Tuple[Optional[Callable[[str], int]], str]:
    if args.hf_tokenizer or args.tokenizer_file:
        try:
            from tokenizers import Tokenizer  # type: ignore
        except ImportError:
            sys.exit("Install the tokenizer library first: pip install tokenizers")
        tok = (Tokenizer.from_file(args.tokenizer_file) if args.tokenizer_file
               else Tokenizer.from_pretrained(args.hf_tokenizer))
        label = args.tokenizer_file or f"hf:{args.hf_tokenizer}"
        return (lambda s: len(tok.encode(s, add_special_tokens=False).ids)), label
    if args.tiktoken:
        try:
            import tiktoken  # type: ignore
        except ImportError:
            sys.exit("Install the tokenizer library first: pip install tiktoken")
        enc = tiktoken.get_encoding(args.tiktoken)
        return (lambda s: len(enc.encode(s, disallowed_special=()))), f"tiktoken:{args.tiktoken}"
    return None, "none"


# ----------------------------------------------------------------------------- helpers

def usage_of(resp: dict) -> Optional[dict]:
    u = resp.get("usage")
    return u if isinstance(u, dict) else None


def is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def message_text(resp: dict) -> Tuple[str, str]:
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    if isinstance(content, list):  # content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if not isinstance(reasoning, str):
        reasoning = ""
    return content, reasoning


def find_cost(obj: Any, depth: int = 0) -> Optional[Tuple[str, float]]:
    """Looks for a provider-reported per-request cost field (e.g. _devup.cost_dzd, usage.cost)."""
    if depth > 3 or not isinstance(obj, dict):
        return None
    for k, v in obj.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and "cost" in k.lower():
            return k, float(v)
    for k, v in obj.items():
        if isinstance(v, dict) and k != "choices":
            hit = find_cost(v, depth + 1)
            if hit:
                return f"{k}.{hit[0]}", hit[1]
    return None


def chat_body(model: str, content: str, max_tokens: int, **extra) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}
    if extra.get("temperature") is not None:
        body["temperature"] = extra["temperature"]
    if extra.get("stream"):
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


# ----------------------------------------------------------------------------- audit

def run(args) -> List[Check]:
    key = os.environ.get(args.key_env, "")
    if not key:
        sys.exit(f"Set your API key in the {args.key_env} environment variable (it is never printed).")
    c = Client(args.base_url, key, args.timeout)
    count, tok_label = load_tokenizer(args)
    temp = None if args.no_temperature else 0
    results: List[Check] = []
    add = results.append
    model_obj: dict = {}

    # 1. model listed -------------------------------------------------------------------------
    try:
        listing, dt = c.get_json("/models")
        ids = [m.get("id") for m in (listing.get("data") or []) if isinstance(m, dict)]
        if args.model in ids:
            model_obj = next(m for m in listing["data"] if m.get("id") == args.model)
            facts = {k: model_obj.get(k) for k in ("context_window", "max_output_tokens", "owned_by") if k in model_obj}
            add(Check("model listed by /models", PASS, f"{len(ids)} models listed; requested model present {facts or ''}".strip(), facts))
        else:
            add(Check("model listed by /models", FAIL, f"{len(ids)} models listed; '{args.model}' is not among them"))
    except ApiError as e:
        add(Check("model listed by /models", FAIL, f"{e} {e.body}"))
    except Exception as e:  # network, DNS, TLS
        add(Check("model listed by /models", FAIL, f"{type(e).__name__}: {e}"))
        return results

    # 2-3. basic completion: model echo + usage ------------------------------------------------
    base_resp: Optional[dict] = None
    try:
        base_resp, dt = c.post_json("/chat/completions", chat_body(args.model, BASE_PROMPT, 160, temperature=temp))
        echoed = base_resp.get("model")
        if echoed == args.model:
            add(Check("response names the requested model", PASS, f"model = {echoed} ({dt:.1f}s)"))
        elif echoed and (args.model.split("/")[-1].lower() in str(echoed).lower()):
            add(Check("response names the requested model", WARN, f"asked {args.model}, response says {echoed} (a dated snapshot or alias?)"))
        else:
            add(Check("response names the requested model", FAIL, f"asked {args.model}, response says {echoed!r}"))
        u = usage_of(base_resp)
        if not u:
            add(Check("usage returned", FAIL, "no usage object: you cannot audit what you are billed for"))
        else:
            p, cpl, tot = u.get("prompt_tokens"), u.get("completion_tokens"), u.get("total_tokens")
            if all(is_int(x) for x in (p, cpl)) and (tot is None or (is_int(tot) and tot == p + cpl)):
                add(Check("usage returned", PASS, f"prompt={p} completion={cpl} total={tot}", {"prompt": p, "completion": cpl}))
            else:
                add(Check("usage returned", FAIL, f"inconsistent usage: {u}"))
    except ApiError as e:
        add(Check("basic chat completion", FAIL, f"{e} {e.body}"))

    # 4-5. tokenizer delta + hidden prompt overhead --------------------------------------------
    p1 = (usage_of(base_resp) or {}).get("prompt_tokens") if base_resp else None
    if is_int(p1):
        try:
            long_resp, _ = c.post_json("/chat/completions",
                                       chat_body(args.model, BASE_PROMPT + SEP + FILLER, 160, temperature=temp))
            p2 = (usage_of(long_resp) or {}).get("prompt_tokens")
            if not is_int(p2):
                add(Check("prompt counted with the model's tokenizer", FAIL, "second response has no prompt_tokens"))
            else:
                delta = p2 - p1
                if count is None:
                    add(Check("prompt counted with the model's tokenizer", INFO,
                              f"appending the same reference text added {delta} prompt tokens; "
                              f"pass --hf-tokenizer or --tiktoken to compare with the official tokenizer", {"delta": delta}))
                else:
                    expected = count(BASE_PROMPT + SEP + FILLER) - count(BASE_PROMPT)
                    tol = max(3, round(expected * 0.02))
                    diff = delta - expected
                    st = PASS if abs(diff) <= tol else FAIL
                    why = "" if st == PASS else " (a different tokenizer counted your prompt: a different model family, or the prompt was rewritten)"
                    add(Check("prompt counted with the model's tokenizer", st,
                              f"provider delta={delta}, {tok_label} expected={expected}, diff={diff:+d}, tolerance ±{tol}{why}",
                              {"delta": delta, "expected": expected}))
                    overhead = p1 - count(BASE_PROMPT)
                    st2 = PASS if overhead <= args.max_overhead else WARN
                    add(Check("no large hidden text added to your prompt", st2,
                              f"{overhead} prompt tokens beyond your own text (chat template{' + something else' if st2 == WARN else ''}); "
                              f"threshold {args.max_overhead}", {"overhead": overhead}))
        except ApiError as e:
            add(Check("prompt counted with the model's tokenizer", FAIL, f"{e} {e.body}"))
    else:
        add(Check("prompt counted with the model's tokenizer", SKIP, "needs prompt_tokens from the basic completion"))

    # 6. completion tokens vs delivered text --------------------------------------------------
    if base_resp and usage_of(base_resp) and count is not None:
        u = usage_of(base_resp) or {}
        content, reasoning = message_text(base_resp)
        details = u.get("completion_tokens_details") or {}
        hidden_reasoning = is_int(details.get("reasoning_tokens")) and details["reasoning_tokens"] > 0 and not reasoning
        cpl = u.get("completion_tokens")
        if hidden_reasoning:
            add(Check("completion tokens match delivered text", SKIP,
                      f"model reasons privately ({details['reasoning_tokens']} reasoning tokens not returned); cannot be counted from outside"))
        elif is_int(cpl) and (content or reasoning):
            seen = count(content) + (count(reasoning) if reasoning else 0)
            tol = max(3, round(seen * 0.03))
            diff = cpl - seen
            if abs(diff) <= tol:
                st = PASS
            elif diff > max(5, seen * 0.15):
                st = FAIL
            else:
                st = WARN
            add(Check("completion tokens match delivered text", st,
                      f"billed completion={cpl}, text you received={seen} tokens ({tok_label}), diff={diff:+d}, tolerance ±{tol}",
                      {"billed": cpl, "seen": seen}))
        else:
            add(Check("completion tokens match delivered text", SKIP, "no completion text to count"))
    elif count is None:
        add(Check("completion tokens match delivered text", SKIP, "pass --hf-tokenizer or --tiktoken"))

    # 6b. provider-reported cost vs published price (informational) ----------------------------
    if base_resp and usage_of(base_resp):
        cost = find_cost({k: v for k, v in base_resp.items() if k not in ("choices",)})
        pricing = model_obj.get("pricing") if isinstance(model_obj, dict) else None
        if cost:
            msg = f"provider reports {cost[0]} = {cost[1]:.6f} for this request"
            est = _estimate_cost(pricing, usage_of(base_resp) or {})
            if est is not None:
                ratio = cost[1] / est if est else float("inf")
                # 5% relative, or 0.0001 absolute for providers that round each charge to 4 decimals
                st = PASS if (abs(ratio - 1) <= 0.05 or abs(cost[1] - est) <= 0.0001) else WARN
                msg += f"; published price × usage = {est:.6f} (ratio {ratio:.3f})"
                add(Check("charged cost matches published price", st, msg))
            else:
                add(Check("charged cost reported per request", INFO, msg))
        else:
            add(Check("charged cost reported per request", INFO, "no per-request cost field; compare usage with your invoice"))

    # 7. max_tokens enforced -------------------------------------------------------------------
    try:
        r, _ = c.post_json("/chat/completions",
                           chat_body(args.model, "Write a 600-word essay about the history of Algiers.", 16, temperature=temp))
        u = usage_of(r) or {}
        cpl = u.get("completion_tokens")
        fr = ((r.get("choices") or [{}])[0]).get("finish_reason")
        if not is_int(cpl):
            add(Check("max_tokens enforced", SKIP, "no usage in the response, cannot verify"))
        elif cpl <= 16 and fr == "length":
            add(Check("max_tokens enforced", PASS, f"max_tokens=16 → completion={cpl}, finish_reason={fr}"))
        elif cpl <= 16:
            add(Check("max_tokens enforced", WARN, f"completion={cpl} within limit but finish_reason={fr!r}"))
        else:
            add(Check("max_tokens enforced", FAIL, f"max_tokens=16 but completion={cpl}, finish_reason={fr!r}"))
    except ApiError as e:
        add(Check("max_tokens enforced", WARN, f"{e} {e.body}"))

    # 8. streaming ------------------------------------------------------------------------------
    try:
        events, ttfc, total, saw_done = c.post_stream("/chat/completions",
                                                      chat_body(args.model, BASE_PROMPT, 160, temperature=temp, stream=True))
        text = "".join(_delta_text(e) for e in events)
        su = next((e["usage"] for e in reversed(events) if isinstance(e.get("usage"), dict)), None)
        ids = {e.get("id") for e in events if e.get("id")}
        problems = []
        if not text:
            problems.append("no streamed text")
        if not su:
            problems.append("no usage in the stream")
        if len(ids) > 1:
            problems.append(f"{len(ids)} different chunk ids")
        detail = f"{len(events)} events, first text after {ttfc:.2f}s, total {total:.2f}s, usage={su}, [DONE]={'yes' if saw_done else 'no'}"
        add(Check("streaming with usage", FAIL if ("no streamed text" in problems or "no usage in the stream" in problems) else (WARN if problems else PASS),
                  detail + (f"; problems: {', '.join(problems)}" if problems else "")))
    except ApiError as e:
        add(Check("streaming with usage", FAIL, f"{e} {e.body}"))

    # 9. context window (opt-in) ---------------------------------------------------------------
    if args.context_test:
        add(context_test(c, args, count, temp))

    return results


def _estimate_cost(pricing: Any, u: dict) -> Optional[float]:
    if not isinstance(pricing, dict):
        return None
    unit = str(pricing.get("unit", "")).lower().replace(" ", "")
    if not any(s in unit for s in ("1m", "million", "1000000")):
        return None
    try:
        p_in = float(pricing["input"])
        p_out = float(pricing.get("output") or 0)
        p_cached = float(pricing["cached"]) if pricing.get("cached") not in (None, "", "—") else None
    except (KeyError, TypeError, ValueError):
        return None
    pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0) if p_cached is not None else 0
    return ((pt - cached) * p_in + cached * (p_cached or 0) + ct * p_out) / 1_000_000


def context_test(c: Client, args, count, temp) -> Check:
    target = args.context_test
    unit = FILLER
    unit_tokens = count(unit) if count else max(1, len(unit) // 3)
    reps = max(1, target // unit_tokens)
    est = reps * unit_tokens
    if not args.yes:
        return Check("advertised context window served", SKIP,
                     f"would send about {est:,} input tokens; re-run with --yes to accept that cost")
    doc = NEEDLE + "\n\n" + unit * reps
    prompt = doc + "\n\nQuestion: what is the audit passphrase stated at the very beginning? Answer with the passphrase only."
    try:
        r, dt = c.post_json("/chat/completions", chat_body(args.model, prompt, 40, temperature=temp))
    except ApiError as e:
        return Check("advertised context window served", FAIL, f"{e} {e.body}")
    content, _ = message_text(r)
    pt = (usage_of(r) or {}).get("prompt_tokens")
    found = "CEDAR-7341-ALGIERS" in content.upper()
    answer = " ".join(content.split())[:80]
    head = f"sent ≈{est:,} tokens, provider counted prompt={pt}, passphrase {'recalled' if found else 'NOT recalled'} ({dt:.1f}s)"
    if not is_int(pt):
        return Check("advertised context window served", FAIL, head + " — no usage returned")
    if count and pt < int(est * 0.95):
        return Check("advertised context window served", FAIL,
                     head + " — fewer prompt tokens than sent: the prompt was truncated before the model")
    if found:
        return Check("advertised context window served", PASS, head)
    # The whole prompt reached the model and was billed, but the model did not use it.
    # That is a limit of the model's long-context recall (or of truncation further upstream
    # that is invisible from outside), not proof of truncation: report it as a warning.
    return Check("advertised context window served", WARN,
                 head + f"; answer={answer!r}. The full prompt was counted, so it was not cut before the model: "
                 "this model does not reliably use that much context. Re-run with a smaller --context-test to find "
                 "where recall holds, or compare with the same model on another provider.")


# ----------------------------------------------------------------------------- output

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit any OpenAI-compatible AI API with your own key.")
    ap.add_argument("--base-url", required=True, help="e.g. https://api.example.com/v1")
    ap.add_argument("--model", required=True, help="exact model id")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hf-tokenizer", help="Hugging Face repo with tokenizer.json (open-weight models)")
    g.add_argument("--tokenizer-file", help="local tokenizer.json (offline)")
    g.add_argument("--tiktoken", help="tiktoken encoding, e.g. o200k_base for OpenAI GPT-4o/GPT-5 models")
    ap.add_argument("--key-env", default="AI_API_KEY", help="environment variable holding the key (default AI_API_KEY)")
    ap.add_argument("--max-overhead", type=int, default=300, help="hidden prompt tokens tolerated (default 300)")
    ap.add_argument("--context-test", type=int, metavar="TOKENS", help="opt-in needle test at about TOKENS input tokens")
    ap.add_argument("--yes", action="store_true", help="accept the cost of --context-test")
    ap.add_argument("--no-temperature", action="store_true", help="do not send temperature (models that reject it)")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--json-out", help="write the results as JSON to this file")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy code page
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print(f"ai-api-audit {VERSION} · {args.base_url} · {args.model}\n")
    results = run(args)
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"[{r.status:4}] {r.name.ljust(width)}  {r.detail}")
    fails = sum(r.status == FAIL for r in results)
    warns = sum(r.status == WARN for r in results)
    print(f"\n{len(results)} checks · {fails} FAIL · {warns} WARN")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"tool": UA, "base_url": args.base_url, "model": args.model,
                       "results": [r.__dict__ for r in results]}, f, ensure_ascii=False, indent=2)
        print(f"results written to {args.json_out}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
