"""Local OpenAI-compatible mock provider used to test ai_api_audit.py without spending any money.

MODE selects the behaviour:
  honest           correct tokenizer, usage, max_tokens, streaming
  substitute       counts prompts with a different tokenizer (another model family behind the name)
  no_usage         never returns usage
  inflate_output   bills 50% more completion tokens than delivered
  hidden_prompt    adds 500 hidden tokens to every prompt
  truncate         silently keeps only the last 2,000 tokens of the prompt
  ignore_max       ignores max_tokens
  weak_recall      counts the full long prompt but the model fails to recall the passphrase
  wrong_model      answers with another model name
"""
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tokenizers import Tokenizer

MODE = sys.argv[1]
PORT = int(sys.argv[2])
TOK = Tokenizer.from_file(sys.argv[3])
MODEL = "acme/real-model-70b"
TEMPLATE = 8
REPLY = "The Sahara is the largest hot desert on Earth. It covers much of North Africa. Its dunes move with the wind."


def n(s):
    return len(TOK.encode(s, add_special_tokens=False).ids)


def prompt_tokens(text):
    if MODE == "substitute":
        return round(n(text) * 1.3) + TEMPLATE
    if MODE == "hidden_prompt":
        return n(text) + TEMPLATE + 500
    if MODE == "truncate":
        return min(n(text), 2000) + TEMPLATE
    return n(text) + TEMPLATE


def answer(text, max_tokens):
    if "passphrase" in text:
        if MODE in ("truncate", "weak_recall") and n(text) > 2000:
            return "I do not know.", "stop"
        return "CEDAR-7341-ALGIERS", "stop"
    reply = REPLY
    if "600-word essay" in text:
        reply = " ".join(["word"] * 400)
    if MODE != "ignore_max" and n(reply) > max_tokens:
        return " ".join(["word"] * max_tokens), "length"
    return reply, "stop"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer test-key":
            return self._send(401, {"error": {"code": "missing_api_key"}})
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [
                {"id": MODEL, "object": "model", "context_window": 131072, "max_output_tokens": 8192,
                 "pricing": {"unit": "1M Tokens", "currency": "DZD", "input": 100, "output": 300}}]})
        self._send(404, {"error": {"code": "endpoint_not_found"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["messages"][-1]["content"]
        mt = body.get("max_tokens", 1024)
        content, fr = answer(text, mt)
        pt = prompt_tokens(text)
        ct = n(content) + (1 if fr == "stop" else 0)  # end-of-sequence token only when the model stopped by itself
        if MODE == "inflate_output":
            ct = round(ct * 1.5)
        usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
        cid = "chatcmpl-" + str(uuid.uuid4())
        model = "acme/cheap-model-8b" if MODE == "wrong_model" else MODEL
        cost = (pt * 100 + ct * 300) / 1_000_000
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for w in content.split(" "):
                ev = {"id": cid, "object": "chat.completion.chunk", "model": model,
                      "choices": [{"index": 0, "delta": {"content": w + " "}}]}
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            last = {"id": cid, "object": "chat.completion.chunk", "model": model, "choices": []}
            if MODE != "no_usage":
                last["usage"] = usage
            self.wfile.write(f"data: {json.dumps(last)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        resp = {"id": cid, "object": "chat.completion", "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": fr}],
                "_provider": {"request_id": "req_1", "cost_dzd": cost}}
        if MODE != "no_usage":
            resp["usage"] = usage
        self._send(200, resp)


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
