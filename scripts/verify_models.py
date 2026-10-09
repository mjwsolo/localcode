#!/usr/bin/env python3
"""Local shipping-server gate. No downloads, remote runner, or project writes.

Run from the candidate checkout: python scripts/verify_models.py --strict
Receipts stay local by default. --publish-status records the result on the exact
clean commit in GitHub; it never connects GitHub Actions to this Mac.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import threading
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

if not __debug__:
    raise RuntimeError("Model verification requires assertions; do not run with python -O")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from localcode import models_catalog as catalog
from localcode.ui import auth
from localcode.ui.server_cmd import server_command

REQUIRED_KEYS = {"gemma-12b", "qwen", "qwen38", "diffusiongemma", "north-mini-code", "muse-glimmer", "kolibri", "openjev"}
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Get weather in a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_stream(f)


def _hash_stream(f) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def environment(values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Server:
    def __init__(self, choice, work: Path, *, turbo=False):
        self.choice, self.work, self.turbo = choice, work, turbo
        self.key = secrets.token_urlsafe(32)
        self.proc = None
        self.log = None

    def __enter__(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.base = f"http://127.0.0.1:{self.port}"
        env = {auth.KEY_ENV: self.key, "LOCALCODE_AGENT_RUN_DIR": str(self.work / "run"), "LOCALCODE_INTERNAL_THINKING_MODE": "off"}
        if self.turbo:
            env.update(LOCALCODE_KV_CACHE_TYPE_K="q8_0", LOCALCODE_KV_CACHE_TYPE_V="turbo4")
        with environment(env):
            self.command = server_command(str(self.choice.local_path), self.port, self.choice.key)
        # A user's custom binary must not redirect a test of the shipped binary.
        self.command[0] = str(ROOT / "src/localcode/bin/llama-server")
        self.log = (self.work / "server.log").open("wb")
        try:
            self.proc = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT, env={k: v for k, v in os.environ.items() if k not in {"GH_TOKEN", "GITHUB_TOKEN"}})
            deadline = time.monotonic() + 420
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError("shipping server exited during model load")
                try:
                    self.request("/health", timeout=2)
                    return self
                except (OSError, ValueError):
                    time.sleep(1)
            raise TimeoutError("model load exceeded 420 seconds")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def request(self, endpoint, body=None, *, authenticated=True, timeout=240):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers.update(auth.server_headers(self.key))
        req = urllib.request.Request(self.base + endpoint, data=None if body is None else json.dumps(body).encode(), headers=headers)
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as res:
            return json.load(res)

    def __exit__(self, *exc):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=15)
        if self.log is not None:
            self.log.close()


def validate_tool(message):
    calls = message.get("tool_calls") or []
    assert calls, "tool request returned no tool call"
    call = calls[0]
    assert call["function"]["name"] == "get_weather", "wrong tool selected"
    args = json.loads(call["function"]["arguments"])
    assert isinstance(args.get("city"), str) and "paris" in args["city"].lower(), "wrong tool arguments"
    assert call.get("id"), "tool call has no correlation ID"
    return call


def validate_decisions(data):
    answers = data["answers"]
    probs = answers["route"]["probabilities"]
    assert set(probs) == {"billing", "shipping", "technical"}
    assert all(isinstance(p, (float, int)) and math.isfinite(p) and 0 <= p <= 1 for p in probs.values())
    assert abs(sum(probs.values()) - 1) < 0.01
    assert answers["route"]["choice"] in probs
    assert 0 <= answers["angry"]["noul"] <= 1
    assert 0 <= answers["urgency"]["score"] <= 2


def runtime_turn(server):
    """Drive the actual bundled OpenCode client and plugin in a disposable project."""
    from localcode.ui.launch import write_config, runtime_env
    config = server.work / "runtime.json"
    with environment({auth.KEY_ENV: server.key}):
        ctx = int(server.command[server.command.index("--ctx-size") + 1])
        write_config(config, port=server.port, ctx=ctx, alias=server.choice.key)
    env = {**{k: v for k, v in runtime_env(config, 1).items() if k not in {"GH_TOKEN", "GITHUB_TOKEN"}}, "OPENCODE_CONFIG": str(config), "XDG_DATA_HOME": str(server.work / "data"), "XDG_CONFIG_HOME": str(server.work / "config"), "XDG_CACHE_HOME": str(server.work / "cache"), "LOCALCODE_CONTROL_PORT": "1", "LOCALCODE_AUTO_CHECK": "0"}
    result = subprocess.run([str(ROOT / "src/localcode/bin/localcode-ui"), "run", "--format", "json", "Reply with exactly OK."], cwd=server.work, env=env, capture_output=True, text=True, timeout=240)
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert result.returncode == 0, "bundled runtime failed"
    assert not any(e.get("type") == "error" for e in events), "bundled runtime returned an error"
    assert any(e.get("type") == "text" and e.get("part", {}).get("text", "").strip() for e in events), "bundled runtime returned no text"


def decision_control(server, body):
    """Verify the shipped authenticated bridge, not only the model endpoint."""
    from localcode.decision import post, validate_response
    from localcode.ui.supervisor import Supervisor, make_handler
    token = secrets.token_urlsafe(32)
    sup = types.SimpleNamespace(control_token=token, control_port=0,
                                current=server.choice.filename.removesuffix(".gguf"),
                                proc=server.proc, port=server.port, state={"state": "ready"})
    sup.decision = lambda request: Supervisor.decision(sup, request)
    with environment({auth.KEY_ENV: server.key}):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(sup))
        sup.control_port = httpd.server_address[1]
        worker = threading.Thread(target=httpd.serve_forever, daemon=True)
        worker.start()
        try:
            url = f"http://127.0.0.1:{sup.control_port}/decision"
            result = post(url, body, auth.control_headers(token))
            validate_response(result, body)
            assert result["model"] == sup.current
        finally:
            httpd.shutdown()
            httpd.server_close()
            worker.join(timeout=5)


def verify(server, *, runtime=False):
    messages = [{"role": "system", "content": "You are a helpful assistant.\n\nAnswer briefly."}, {"role": "user", "content": "Reply with exactly OK."}]
    body = {"model": server.choice.key, "messages": messages, "temperature": 0, "max_tokens": 8192, "cache_prompt": True}
    # Authentication is checked on inference, not the intentionally public health endpoint.
    try:
        server.request("/v1/chat/completions", body, authenticated=False)
    except urllib.error.HTTPError as exc:
        assert exc.code == 401, f"unauthenticated request returned {exc.code}"
    else:
        raise AssertionError("unauthenticated inference was accepted")
    assert server.request("/apply-template", {"messages": messages})["prompt"]
    message = server.request("/v1/chat/completions", body)["choices"][0]["message"]
    assert (message.get("content") or "").strip(), "empty chat response"
    checks = ["auth", "template", "chat"]
    if server.choice.key == "openjev":
        decision_body = {"state": "The customer was charged twice.", "questions": {"route": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": None, "shipping": None, "technical": None}}, "angry": {"type": "noul", "instructions": "Is the customer angry?"}, "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}}}
        data = server.request("/v1/systemone", decision_body)
        validate_decisions(data)
        checks.append("systemone")
        decision_control(server, decision_body)
        checks.append("decision-control")
    else:
        body["messages"][-1]["content"] = "What is the weather in Paris? Use get_weather."
        body["tools"] = [TOOL]
        tool_message = server.request("/v1/chat/completions", body)["choices"][0]["message"]
        call = validate_tool(tool_message)
        body["messages"] += [tool_message, {"role": "tool", "tool_call_id": call["id"], "content": "Paris weather: sunny, 20 Celsius."}]
        final = server.request("/v1/chat/completions", body)["choices"][0]["message"]
        assert (final.get("content") or "").strip(), "empty response after tool result"
        checks.append("tool-loop")
    drafter = getattr(server.choice, "drafter", None)
    if drafter is not None:
        # The shipping configuration includes the vendor drafter: it must be on
        # disk (exact size) and the launcher must have loaded it.
        assert drafter.local_path.is_file() and drafter.local_path.stat().st_size == drafter.size_bytes, "drafter missing or incomplete"
        assert "--model-draft" in server.command and drafter.spec_type in server.command, "drafter not launched"
        timings = server.request("/v1/chat/completions", {**body, "messages": [body["messages"][0], {"role": "user", "content": "Write a Python function that reverses a string. No explanation."}], "max_tokens": 120}).get("timings", {})
        assert timings.get("draft_n", 0) > 0 and timings.get("draft_n_accepted", 0) > 0, "drafter produced no accepted drafts"
        checks.append("drafter")
    if runtime:
        runtime_turn(server)
        checks.append("runtime")
    return checks


def status(commit, state, description):
    # gh uses local credentials; GitHub tokens are not passed to inference subprocesses.
    subprocess.run(["gh", "api", "repos/mjwsolo/localcode/statuses/" + commit, "-f", "state=" + state, "-f", "context=local model gate", "-f", "description=" + description], check=True, stdout=subprocess.DEVNULL)


def validate_receipt(receipt, commit, hashes, keys):
    assert receipt.get("success") is True and receipt.get("strict") is True
    assert receipt.get("dirty") is False and receipt.get("commit") == commit
    assert receipt.get("binaries") == hashes, "candidate binaries changed since verification"
    normal = {r["key"]: set(r["checks"]) for r in receipt["models"] if not r["turbo4"]}
    assert set(normal) == keys and REQUIRED_KEYS <= keys, "model coverage changed"
    for key, checks in normal.items():
        expected = {"auth", "template", "chat", "systemone" if key == "openjev" else "tool-loop"}
        if key in {"qwen38", "openjev", "diffusiongemma"}:
            expected.add("runtime")
        if key == "openjev":
            expected.add("decision-control")
        assert expected <= checks, "missing checks for " + key
    assert any(r["key"] == "qwen" and r["turbo4"] and "tool-loop" in r["checks"] for r in receipt["models"]), "turbo4 not verified"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", help="catalog keys (default: every downloaded catalog model)")
    p.add_argument("--strict", action="store_true", help="fail when required shipping configurations are absent")
    p.add_argument("--receipt", type=Path, default=ROOT / ".localcode-gate/receipt.json")
    p.add_argument("--publish-status", action="store_true", help="record strict gate status for the exact clean commit")
    p.add_argument("--publish-receipt", action="store_true", help="publish a successful strict receipt after pushing its exact commit")
    args = p.parse_args(argv)
    choices = [c for c in catalog.CHOICES if c.local_path.is_file() and (not args.models or c.key in args.models)]
    expected = set(args.models or []) | (REQUIRED_KEYS if args.strict else set())
    missing = expected - {c.key for c in choices}
    if missing or not choices:
        p.error("required models missing: " + ", ".join(sorted(missing or {"any downloaded model"})))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
    if args.publish_receipt and (dirty or args.models or args.publish_status):
        p.error("receipt publication requires a clean full checkout, without --models or --publish-status")
    if args.publish_receipt:
        hashes = {name: sha256(ROOT / "src/localcode/bin" / name) for name in ["llama-server", "localcode-ui"]}
        validate_receipt(json.loads(args.receipt.read_text()), commit, hashes, {c.key for c in choices})
        status(commit, "success", "Strict local shipping launcher and runtime gate passed")
        return 0
    if args.publish_status and (not args.strict or args.models or dirty):
        p.error("publishing requires --strict, the full model set, and a clean tracked checkout")
    if args.publish_status:
        status(commit, "pending", "Local shipping launcher, inference and runtime verification running")
    receipt = {"commit": commit, "dirty": dirty, "strict": args.strict, "binaries": {name: sha256(ROOT / "src/localcode/bin" / name) for name in ["llama-server", "localcode-ui"]}, "models": [], "success": False}
    try:
        runs = [(c, False) for c in choices]
        turbo_choice = next((c for c in choices if c.key == "qwen"), None)
        if turbo_choice:
            runs.append((turbo_choice, True))
        for choice, turbo in runs:
            label = choice.key + ("/turbo4" if turbo else "")
            print("VERIFY " + label, flush=True)
            with tempfile.TemporaryDirectory(prefix="localcode-gate-") as tmp:
                with Server(choice, Path(tmp), turbo=turbo) as server:
                    checks = verify(server, runtime=choice.key in {"qwen38", "openjev", "diffusiongemma"} and not turbo)
            receipt["models"].append({"key": choice.key, "turbo4": turbo, "checks": checks})
            print("PASS " + label + " " + ", ".join(checks), flush=True)
        receipt["success"] = True
    except Exception as exc:
        print("FAIL " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
    finally:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        if args.publish_status:
            status(commit, "success" if receipt["success"] else "failure", "Local shipping launcher and runtime gate " + ("passed" if receipt["success"] else "failed"))
    return 0 if receipt["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
