"""Prompt warm-up: slot-file token parsing, capture merging, settle detection, store."""
import json
import struct

import pytest

from localcode.ui import warmup


def _slot_file(tmp_path, tokens, packed=True):
    if packed:
        entries = [0xFFFFFFFF, 1, len(tokens), *tokens, 0]  # marker, version, count, ids, no media keys
    else:
        entries = list(tokens)
    p = tmp_path / "warm.bin"
    with p.open("wb") as f:
        f.write(struct.pack("<III", 0x67677371, 2, len(entries)))
        f.write(struct.pack(f"<{len(entries)}I", *entries))
        f.write(b"\0" * 64)  # KV state follows; irrelevant here
    return p


def test_parse_packed_and_plain_slot_files(tmp_path):
    toks = [2, 105, 9731, 107, 3048]
    assert warmup.parse_slot_tokens(_slot_file(tmp_path, toks)) == toks
    assert warmup.parse_slot_tokens(_slot_file(tmp_path, toks, packed=False)) == toks


def test_parse_rejects_foreign_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"GGUF" + b"\0" * 32)
    with pytest.raises(ValueError):
        warmup.parse_slot_tokens(p)


def test_merge_keeps_shared_prefix_and_drops_user_message():
    prefix = list(range(1000))
    s1 = prefix + [7, 7, 7]          # session 1: prefix + "Say hi"
    s2 = prefix + [9, 9, 9, 9]       # session 2: prefix + another message
    assert warmup.merge_tokens(None, s1) == s1
    assert warmup.merge_tokens(s1, s2) == prefix


def test_merge_replaces_when_captures_barely_overlap():
    old = list(range(1000))
    new = list(range(300)) + list(range(5000, 6000))   # date line moved: only 300 shared
    assert warmup.merge_tokens(old, new) == new


def test_merge_caps_length():
    new = list(range(warmup.MAX_TOKENS + 500))
    assert len(warmup.merge_tokens(None, new)) == warmup.MAX_TOKENS


def _slot(n, task, processing=False, sid=0):
    return {"id": sid, "is_processing": processing, "n_prompt_tokens": n, "id_task": task}


def test_tracker_waits_for_settled_agent_sized_prompt():
    t = warmup.CaptureTracker(baseline_task=3, settle_s=3.0)
    assert t.observe([_slot(5000, 3)], 0.0) is None            # the warm-up replay itself
    assert t.observe([_slot(560, 4)], 1.0) is None             # title request finished
    assert t.observe([_slot(3600, 7, processing=True)], 2.0) is None
    assert t.observe([_slot(3600, 7)], 3.0) is None            # idle, clock starts
    assert t.observe([_slot(3600, 7)], 5.0) is None
    assert t.observe([_slot(3600, 7)], 6.1) == 0               # settled 3 s on the same task


def test_tracker_resets_when_a_new_task_lands():
    t = warmup.CaptureTracker(settle_s=3.0)
    assert t.observe([_slot(3600, 7)], 0.0) is None
    assert t.observe([_slot(3600, 7, processing=True)], 2.0) is None
    assert t.observe([_slot(4100, 9)], 2.5) is None
    assert t.observe([_slot(4100, 9)], 5.4) is None
    assert t.observe([_slot(4100, 9)], 5.6) == 0


def test_tracker_ignores_small_prompts():
    t = warmup.CaptureTracker(settle_s=0.0)
    assert t.observe([_slot(400, 2)], 0.0) is None
    assert t.observe([_slot(400, 2)], 9.0) is None


def test_store_roundtrip_and_fork_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALCODE_AGENT_RUN_DIR", str(tmp_path))
    toks = list(range(700))
    warmup.store("gemma-4-12b-it-UD-Q4_K_XL", toks)
    assert warmup.load("gemma-4-12b-it-UD-Q4_K_XL") == toks
    f = tmp_path / "warm" / "gemma-4-12b-it-UD-Q4_K_XL.json"
    data = json.loads(f.read_text()); data["fork"] = "other"; f.write_text(json.dumps(data))
    assert warmup.load("gemma-4-12b-it-UD-Q4_K_XL") is None
    warmup.forget("gemma-4-12b-it-UD-Q4_K_XL")
    assert not f.exists()


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("LOCALCODE_PROMPT_WARMUP", "0")
    assert not warmup.enabled()
    monkeypatch.delenv("LOCALCODE_PROMPT_WARMUP")
    assert warmup.enabled()


def test_server_command_carries_slot_save_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALCODE_AGENT_RUN_DIR", str(tmp_path))
    from localcode.ui.server_cmd import server_command
    cmd = server_command(str(tmp_path / "x.gguf"), 8123, "x")
    assert cmd[cmd.index("--slot-save-path") + 1] == str(tmp_path / "slots")
    monkeypatch.setenv("LOCALCODE_PROMPT_WARMUP", "0")
    assert "--slot-save-path" not in server_command(str(tmp_path / "x.gguf"), 8123, "x")


def test_replay_cancel_closes_the_socket_promptly():
    """Closing the client socket is the cancel: llama-server drops a task whose client left."""
    import socket
    import threading
    import time

    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    def stall():  # accept, read the request, never answer (a prefill in progress)
        c, _ = srv.accept(); c.recv(65536); time.sleep(5); c.close()
    threading.Thread(target=stall, daemon=True).start()
    rp = warmup.Replay(port, timeout=30)
    t = threading.Timer(0.3, rp.cancel); t.start()
    t0 = time.time()
    with pytest.raises(Exception):
        rp.run(list(range(600)))
    assert rp.cancelled and time.time() - t0 < 3
