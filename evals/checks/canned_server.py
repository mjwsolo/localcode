#!/usr/bin/env python3
"""A canned OpenAI-compatible model server for localcode Checks.

No model. Every /v1/chat/completions body is written to <outdir>/req-NN.json,
then answered from a fixed script so a whole multi-turn conversation runs in
seconds:
  - a request with no tools (title generation, summaries) -> a short text
  - a compaction request (it carries the agent's tools) -> a short summary
  - first request of a turn (last message is from the user) -> one tool call
      turn 1: read <workspace>/README.md (CANNED_PARALLEL=1: README.md and a.py in one response)
      turn 2: todowrite with two items
  - a request whose last message is a tool result -> a short text answer
Also serves GET /slots (CANNED_SLOTS entries), /health, /v1/models, /props.
Usage: canned_server.py <port> <outdir>
"""
import http.server, json, os, sys, time, uuid

from http_fixture import LoopbackHTTPServer

PORT = int(sys.argv[1]); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
SLOTS = int(os.environ.get("CANNED_SLOTS", "1")); WS = os.environ.get("CANNED_WORKSPACE", "/tmp")
N = 0
FORCED = False

def script(body):
    msgs = body.get("messages", [])
    tools = body.get("tools") or []
    if not tools:
        return {"text": "Canned title"}
    last = msgs[-1] if msgs else {}
    if last.get("role") == "user" and "summarize the conversation" in json.dumps(last.get("content")).lower():
        # A compaction request: it shares the agent's tools so the cached prefix is reused.
        return {"text": "## Objective\n- Answer questions about the demo project.\n\n## Next Move\n1. (none)"}
    if last.get("role") == "tool":
        return {"text": "Done. The file was read and the plan is set."}
    users = sum(1 for m in msgs if m.get("role") == "user")
    names = {t.get("function", {}).get("name") for t in tools}
    if users <= 1 and "read" in names:
        if os.environ.get("CANNED_PARALLEL") == "1":
            return {"tools": [("read", {"path": os.path.join(WS, "README.md")}), ("read", {"path": os.path.join(WS, "a.py")})]}
        return {"tool": "read", "args": {"path": os.path.join(WS, "README.md")}}
    if "todowrite" in names:
        return {"tool": "todowrite", "args": {"todos": [
            {"id": "1", "content": "Add a docstring to add()", "status": "in_progress", "priority": "medium"},
            {"id": "2", "content": "Run the tests", "status": "pending", "priority": "medium"}]}}
    return {"text": "Canned answer."}

def chunks(rid, model, action):
    global FORCED
    prompt_tokens = 100
    if "tools" not in action and "tool" in action:
        action = {**action, "tools": [(action["tool"], action["args"])]}
    if "tools" in action and not FORCED and os.environ.get("CANNED_FORCE_COMPACTION") == "1":
        prompt_tokens = 32700
        FORCED = True
    base = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
    if "text" in action:
        yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": action["text"]}, "finish_reason": None}]}
        yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 8, "total_tokens": 108}}
        return
    for index, (name, args) in enumerate(action["tools"]):
        call = {"index": index, "id": "call_" + uuid.uuid4().hex[:8], "type": "function", "function": {"name": name, "arguments": ""}}
        yield {**base, "choices": [{"index": 0, "delta": {**({"role": "assistant"} if index == 0 else {}), "tool_calls": [call]}, "finish_reason": None}]}
        yield {**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "function": {"arguments": json.dumps(args)}}]}, "finish_reason": None}]}
    yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 20, "total_tokens": prompt_tokens + 20}}

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        data = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/slots": return self._json([{"id": i, "n_ctx": 32768, "is_processing": False} for i in range(SLOTS)])
        if p == "/health": return self._json({"status": "ok"})
        if p == "/v1/models": return self._json({"object": "list", "data": [{"id": "canned-model", "object": "model"}]})
        if p == "/props": return self._json({"total_slots": SLOTS, "default_generation_settings": {"n_ctx": 32768}})
        return self._json({"error": "not found"}, 404)
    def do_POST(self):
        global N
        ln = int(self.headers.get("Content-Length") or 0); raw = self.rfile.read(ln) if ln else b"{}"
        if not self.path.endswith("/chat/completions"):
            return self._json({"ok": True})
        body = json.loads(raw or b"{}"); N += 1
        with open(os.path.join(OUT, f"req-{N:02d}.json"), "wb") as f: f.write(raw)
        action = script(body); rid = "chatcmpl-" + uuid.uuid4().hex[:12]; model = body.get("model", "canned-model")
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
            for c in chunks(rid, model, action):
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n"); self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            return
        msg = {"role": "assistant", "content": action.get("text")}
        calls = action.get("tools") or ([(action["tool"], action["args"])] if "tool" in action else [])
        if calls:
            msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_" + uuid.uuid4().hex[:8], "type": "function", "function": {"name": name, "arguments": json.dumps(args)}} for name, args in calls]}
        self._json({"id": rid, "object": "chat.completion", "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})

LoopbackHTTPServer(("127.0.0.1", PORT), H).serve_forever()
