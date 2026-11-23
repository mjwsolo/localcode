#!/usr/bin/env python3
"""A deterministic mock for the e2e screenshot suite: one loopback port that is
both an OpenAI-compatible model server and the supervisor's control API.

No model, no llama-server. Replies are fixed so screenshots are stable:
  - requests without tools (title generation)   -> "E2E title"
  - a user message containing "run ls"          -> one scripted shell tool call
  - a user message containing "read outside"    -> a read of /tmp/lce-outside/notes.txt (asks permission)
  - a user message containing "make a plan"     -> three todowrite calls in one turn, each
                                                   marking the next step done (2 s apart)
  - a tool result as the last message           -> "The command finished."
  - anything else                               -> a fixed two-line reply
Usage: mock_server.py <port> [--no-model]
"""
import http.server
import json
import sys
import time
import uuid
from http.server import ThreadingHTTPServer
from socketserver import TCPServer
from urllib.parse import parse_qs, urlparse

PORT = int(sys.argv[1])
NO_MODEL = "--no-model" in sys.argv
# --vision=off-available: no projector loaded, one can be downloaded (0.9 GB)
# --vision=off-unavailable: no projector loaded and none exists for this model
# --vision=on: images are readable. Default is off-unavailable.
VISION = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--vision=")), "off-unavailable")
# Every POST path and chat request the mock saw, for tests (GET /_log).
LOG = []
ALIAS = "e2e-model"

# A background download started from the picker: it runs for DOWNLOAD_SECONDS.
DOWNLOAD = {}
DOWNLOAD_SECONDS = 12


def download():
    if not DOWNLOAD:
        return None
    if DOWNLOAD["state"] == "downloading":
        done = (time.time() - DOWNLOAD["t0"]) / DOWNLOAD_SECONDS
        if done >= 1:
            DOWNLOAD.update(state="done", pct=None, eta_s=None, detail="downloaded")
        else:
            DOWNLOAD["pct"] = int(done * 100)
            DOWNLOAD["eta_s"] = 180  # fixed so screens stay stable
    return {k: v for k, v in DOWNLOAD.items() if k != "t0"}


REPLY = "Hello from the mock model.\nThis reply is fixed so screenshots stay stable."


def _group(key, name, maker, lic, repo, rec, installed):
    return {"key": key, "display_name": name, "maker": maker, "license": lic, "hf_repo": repo,
            "recommended": rec, "current": False, "loaded": False, "installed_count": installed,
            "downloading": False, "pct": None}


GROUPS = [
    _group("alpha", "Alpha 4 12B", "Acme", "apache-2.0", "acme/alpha-gguf", True, 1),
    _group("beta", "Beta 3 8B", "Thinking Machines", "mit", "globex/beta-gguf", False, 0),
    _group("gamma", "Gamma 2 30B", "Initech", "apache-2.0", "initech/gamma-gguf", False, 0),
]


def quants(key):
    g = next((g for g in GROUPS if g["key"] == key), GROUPS[0])
    rows = [
        ("Q3_K_M", 5.1, "fits", 41.0, False, False, True, 21.4),
        ("Q4_K_M", 6.9, "fits", 33.5, True, True, True, 19.6),
        ("Q6_K", 9.8, "fits", 24.0, False, False, True, 16.7),
        ("Q8_0", 12.7, "fits", 19.5, False, False, False, 6.8),
    ]
    return {"group": g["key"], "display_name": g["display_name"], "maker": g["maker"],
            "license": g["license"], "ram_gb": 32, "listing": False, "drafter_size_gb": 0.0,
            "vision_size_gb": 0.0, "largest_fit_idx": 3,
            "quants": [{"filename": f"{key}-{label}.gguf", "alias": f"{key}-{label}", "label": label,
                        "size_gb": size, "fit": fit, "tok_s": tok, "recommended": rec,
                        "comfortable": comfy, "free_gb": free,
                        "downloaded": dl or (DOWNLOAD.get("state") == "done" and DOWNLOAD.get("filename") == f"{key}-{label}.gguf"),
                        "current": False, "loaded": False,
                        "downloading": DOWNLOAD.get("state") == "downloading" and DOWNLOAD.get("filename") == f"{key}-{label}.gguf",
                        "pct": DOWNLOAD.get("pct")}
                       for label, size, fit, tok, rec, dl, comfy, free in rows]}


def status():
    return {"state": "idle" if NO_MODEL else "ready", "model": None if NO_MODEL else ALIAS,
            "detail": "", "pct": None, "server_running": not NO_MODEL, "server_ready": not NO_MODEL,
            "version": "0.0.0", "process_id": 1, "warmup": "idle",
            "current": None if NO_MODEL else ALIAS, "port": PORT, "inference_port": PORT,
            "ctx": 32768, "group": None, "filename": None, "vision": VISION == "on",
            "vision_available": VISION == "off-available", "vision_size_gb": 0.9 if VISION == "off-available" else 0.0, "download": download()}


PLAN_STEPS = ["Alpha step", "Beta step", "Gamma step"]


def plan(done):
    """Steps before `done` are completed, step `done` is in progress."""
    return [{"content": c, "status": "completed" if i < done else "in_progress" if i == done else "pending",
             "priority": "medium"} for i, c in enumerate(PLAN_STEPS)]


def script(body):
    msgs = body.get("messages", [])
    if not body.get("tools"):
        return {"text": "E2E title"}
    last = msgs[-1] if msgs else {}
    users = [m for m in msgs if m.get("role") == "user"]
    if users and "make a plan" in json.dumps(users[-1].get("content")).lower():
        start = max(i for i, m in enumerate(msgs) if m.get("role") == "user")
        done = sum(1 for m in msgs[start:] if m.get("role") == "tool")
        if done < len(PLAN_STEPS):
            return {"tool": "todowrite", "args": {"todos": plan(done)}, "delay": 2 if done else 0}
        return {"text": "Plan finished.", "delay": 2}
    if last.get("role") == "tool":
        return {"text": "The command finished."}
    if "read outside" in json.dumps(last.get("content")).lower():
        return {"tool": "read", "args": {"path": "/tmp/lce-outside/notes.txt"}}
    if "run ls" in json.dumps(last.get("content")).lower():
        return {"tool": "shell", "args": {"command": "ls", "description": "List files"}}
    return {"text": REPLY}


def chunks(rid, model, action):
    base = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
    usage = {"prompt_tokens": 100, "completion_tokens": 8, "total_tokens": 108}
    if "text" in action:
        yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": action["text"]}, "finish_reason": None}]}
        yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}
        return
    call = {"index": 0, "id": "call_" + uuid.uuid4().hex[:8], "type": "function",
            "function": {"name": action["tool"], "arguments": ""}}
    yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [call]}, "finish_reason": None}]}
    yield {**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": json.dumps(action["args"])}}]}, "finish_reason": None}]}
    yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": usage}


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p == "/health":
            return self._json({"status": "ok"})
        if p == "/v1/models":
            return self._json({"object": "list", "data": [{"id": ALIAS, "object": "model"}]})
        if p == "/props":
            return self._json({"total_slots": 1, "default_generation_settings": {"n_ctx": 32768}})
        if p == "/slots":
            return self._json([{"id": 0, "n_ctx": 32768, "is_processing": False}])
        if p == "/status":
            return self._json(status())
        if p == "/catalog":
            return self._json({"ram_gb": 32, "current": None if NO_MODEL else ALIAS, "groups": GROUPS,
                               "models_dir": "~/models"})
        if p == "/quants":
            return self._json(quants(parse_qs(u.query).get("group", ["alpha"])[0]))
        if p == "/_log":
            return self._json(LOG)
        if p == "/voice/status":
            return self._json({"available": False})
        if p == "/progress":
            return self._json({"phase": "idle"})
        if p == "/models_dir":
            return self._json({"path": "~/models", "exists": True})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        ln = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(ln) if ln else b"{}"
        LOG.append(self.path)
        if self.path == "/vision/install":
            global VISION
            VISION = "on"
            return self._json({"ok": True})
        if self.path == "/select":
            req = json.loads(raw or b"{}")
            name = req.get("filename", "")
            alias = name.removesuffix(".gguf")
            if name.endswith("Q4_K_M.gguf"):  # the downloaded quant
                return self._json({"ok": True, "model": alias, "state": "loading"})
            DOWNLOAD.clear()
            DOWNLOAD.update(group=req.get("group"), filename=name, model=alias, state="downloading", pct=0, detail="",
                            size_gb=5.1, eta_s=None, t0=time.time())
            return self._json({"ok": True, "model": alias, "state": "downloading", "background": True})
        if self.path == "/cancel" and DOWNLOAD.get("state") == "downloading":
            DOWNLOAD.update(state="cancelled", pct=None)
            return self._json({"ok": True})
        if not self.path.endswith("/chat/completions"):
            return self._json({"ok": True})
        body = json.loads(raw or b"{}")
        action = script(body)
        time.sleep(action.get("delay", 0))
        rid = "chatcmpl-" + uuid.uuid4().hex[:12]
        model = body.get("model", ALIAS)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for c in chunks(rid, model, action):
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        msg = {"role": "assistant", "content": action.get("text")}
        if "tool" in action:
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                 "function": {"name": action["tool"], "arguments": json.dumps(action["args"])}}]}
        self._json({"id": rid, "object": "chat.completion", "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if "tool" in action else "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})


class Loopback(ThreadingHTTPServer):
    def server_bind(self):
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


Loopback(("127.0.0.1", PORT), H).serve_forever()
