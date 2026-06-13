"""Harder A/B: SYSTEM_PROMPT vs SYSTEM_PROMPT_V2 with REAL headroom.

The trivial 8-task bench hit 100% on both prompts on every model — a ceiling
that can't distinguish prompt quality. This eval adds two harder categories
so differences can actually show:

  * HARD_TASKS  — LeetCode easy/medium code-gen (decode_string, edit_distance,
    coin_change, ...). Quantized small models fail some → headroom. Single
    turn, inline code, scored by hidden asserts. (Tests instruction-following
    + correctness.)
  * AGENTIC_TASKS — "create a file containing function f" run WITH real tools.
    Scores what the agent prompt is actually FOR: did the model ACT via a
    tool call (not narrate), and is the delivered code correct + complete?
    (Tests act-don't-narrate + tool reliability + complete-code rules.)

Reuses the fixed scorer (_score evaluates the assert EXPRESSION, not the
(expr,label) tuple) and server plumbing from bench_prompt_variants.

Usage:
    PYTHONPATH=src:scripts python scripts/bench_prompt_hard.py [model] [k]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localcode import models_catalog as catalog
from localcode.agent.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_V2
from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway
from localcode.toolkit import LocalCodeToolkit

from bench_diffusion_quants import extract_python_code  # noqa: E402
from bench_prompt_variants import _fill, _score, _wait_health, CODE_ONLY  # noqa: E402

PORT = 8195

VARIANTS = {"current": SYSTEM_PROMPT, "v2_lean": SYSTEM_PROMPT_V2}

# ── Hard single-turn code-gen (headroom on quantized small models) ───
HARD_TASKS = [
    {"id": "roman_to_int",
     "prompt": "Write a Python function `roman_to_int(s)` that converts an uppercase Roman numeral string to an integer.",
     "asserts": [("roman_to_int('III') == 3", ""), ("roman_to_int('IV') == 4", ""),
                 ("roman_to_int('IX') == 9", ""), ("roman_to_int('LVIII') == 58", ""),
                 ("roman_to_int('MCMXCIV') == 1994", "")]},
    {"id": "valid_parens",
     "prompt": "Write a Python function `is_valid(s)` that returns True iff the brackets in s (containing only the characters ()[]{}) are balanced and correctly nested.",
     "asserts": [("is_valid('()') == True", ""), ("is_valid('()[]{}') == True", ""),
                 ("is_valid('(]') == False", ""), ("is_valid('([)]') == False", ""),
                 ("is_valid('{[]}') == True", "")]},
    {"id": "merge_intervals",
     "prompt": "Write a Python function `merge_intervals(intervals)` that merges all overlapping [start, end] intervals and returns the merged list sorted by start.",
     "asserts": [("merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]", ""),
                 ("merge_intervals([[1,4],[4,5]]) == [[1,5]]", ""),
                 ("merge_intervals([[1,4]]) == [[1,4]]", "")]},
    {"id": "coin_change",
     "prompt": "Write a Python function `coin_change(coins, amount)` that returns the fewest number of coins needed to make up amount, or -1 if impossible.",
     "asserts": [("coin_change([1,2,5], 11) == 3", ""), ("coin_change([2], 3) == -1", ""),
                 ("coin_change([1], 0) == 0", ""), ("coin_change([1,5,10,25], 63) == 6", "")]},
    {"id": "decode_string",
     "prompt": "Write a Python function `decode_string(s)` that decodes strings encoded as k[substring], e.g. '3[a]2[bc]' -> 'aaabcbc'. Encodings may be nested.",
     "asserts": [("decode_string('3[a]2[bc]') == 'aaabcbc'", ""),
                 ("decode_string('3[a2[c]]') == 'accaccacc'", ""),
                 ("decode_string('2[abc]3[cd]ef') == 'abcabccdcdcdef'", "")]},
    {"id": "longest_unique",
     "prompt": "Write a Python function `length_of_longest(s)` returning the length of the longest substring of s without repeating characters.",
     "asserts": [("length_of_longest('abcabcbb') == 3", ""), ("length_of_longest('bbbbb') == 1", ""),
                 ("length_of_longest('pwwkew') == 3", ""), ("length_of_longest('') == 0", "")]},
    {"id": "edit_distance",
     "prompt": "Write a Python function `edit_distance(a, b)` returning the Levenshtein edit distance between strings a and b.",
     "asserts": [("edit_distance('horse','ros') == 3", ""), ("edit_distance('intention','execution') == 5", ""),
                 ("edit_distance('','abc') == 3", ""), ("edit_distance('abc','abc') == 0", "")]},
    {"id": "max_subarray",
     "prompt": "Write a Python function `max_subarray(nums)` returning the largest sum of any contiguous non-empty subarray (Kadane's algorithm).",
     "asserts": [("max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6", ""), ("max_subarray([1]) == 1", ""),
                 ("max_subarray([5,4,-1,7,8]) == 23", ""), ("max_subarray([-1,-2,-3]) == -1", "")]},
]

# ── Agentic: create a file via tools; score the delivered code ───────
AGENTIC_TASKS = [
    {"id": "a_gcd", "path": "gcdutil.py",
     "prompt": "Create a Python file `gcdutil.py` containing a function `gcd(a, b)` that returns the greatest common divisor of a and b.",
     "asserts": [("gcd(12,8) == 4", ""), ("gcd(17,5) == 1", ""), ("gcd(0,5) == 5", "")]},
    {"id": "a_count_words", "path": "wc.py",
     "prompt": "Create a Python file `wc.py` containing a function `count_words(s)` that returns the number of whitespace-separated words in s.",
     "asserts": [("count_words('a b c') == 3", ""), ("count_words('') == 0", ""),
                 ("count_words('  hi   there ') == 2", "")]},
    {"id": "a_flatten", "path": "flat.py",
     "prompt": "Create a Python file `flat.py` containing a function `flatten(lst)` that flattens one level of nesting in a list of lists.",
     "asserts": [("flatten([[1,2],[3,4]]) == [1,2,3,4]", ""), ("flatten([[1],[2,3],[]]) == [1,2,3]", "")]},
    {"id": "a_titlecase", "path": "tc.py",
     "prompt": "Create a Python file `tc.py` containing a function `title_case(s)` that capitalizes the first letter of each whitespace-separated word and lowercases the rest.",
     "asserts": [("title_case('hello world') == 'Hello World'", ""),
                 ("title_case('the QUICK fox') == 'The Quick Fox'", "")]},
]


def _code_from_toolcalls(tool_calls: list, visible: str) -> tuple[str, bool]:
    """Return (code, used_a_tool). Prefer write_file content; fall back to
    bash heredoc content, then to inline code in the visible text."""
    used = bool(tool_calls)
    for tc in tool_calls or []:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if name in ("write_file", "append_file") and args.get("content"):
            return args["content"], used
        if name == "bash" and args.get("command"):
            cmd = args["command"]
            # crude heredoc extraction: text between <<'EOF' ... EOF
            if "<<" in cmd and "\n" in cmd:
                body = cmd.split("\n", 1)[1]
                return body.rsplit("\n", 1)[0] if "\n" in body else body, used
    return extract_python_code(visible), used


def main():
    model_file = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-12b-it-UD-Q4_K_XL.gguf"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    model_path = Path(catalog.model_dir()) / model_file
    if not model_path.is_file():
        print(f"SKIP: model not downloaded: {model_path}")
        return

    cfg = RuntimeConfig()
    cfg.provider = "llama_cpp"
    cfg.model = str(model_path)
    gw = LocalCodeRuntimeGateway(cfg)
    tools = LocalCodeToolkit(repo_root=os.getcwd(), config=cfg, app=None).schemas()
    cmd = gw.llama_server_command(str(model_path))
    if "--port" in cmd:
        cmd[cmd.index("--port") + 1] = str(PORT)
    gw.config.base_url = f"http://localhost:{PORT}"
    print(f"Starting server for {model_file} on :{PORT} ...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results: dict = {}
    try:
        _wait_health(proc, f"http://localhost:{PORT}/health")
        print("server up.\n")
        for vname, vprompt in VARIANTS.items():
            sys_msg = {"role": "system", "content": _fill(vprompt)}
            codegen_pass = 0
            codegen_total = 0
            agentic_pass = 0
            agentic_total = 0
            tool_used = 0
            # Hard code-gen
            for t in HARD_TASKS:
                for _ in range(k):
                    content = []
                    for ev in gw.stream_chat_events(
                        [sys_msg, {"role": "user", "content": t["prompt"] + CODE_ONLY}],
                        tools=None, num_predict=512,
                    ):
                        if ev.get("type") == "content":
                            content.append(ev.get("content", ""))
                    code = extract_python_code("".join(content))
                    codegen_total += 1
                    if _score(code, t["asserts"]):
                        codegen_pass += 1
                print(f"[{vname}] code:{t['id']:16} running...")
            # Agentic (real tools)
            for t in AGENTIC_TASKS:
                for _ in range(k):
                    content, tcs = [], []
                    for ev in gw.stream_chat_events(
                        [sys_msg, {"role": "user", "content": t["prompt"]}],
                        tools=tools, num_predict=512,
                    ):
                        if ev.get("type") == "content":
                            content.append(ev.get("content", ""))
                        elif ev.get("type") == "tool_calls":
                            tcs = ev.get("tool_calls", [])
                    code, used = _code_from_toolcalls(tcs, "".join(content))
                    agentic_total += 1
                    if used:
                        tool_used += 1
                    if _score(code, t["asserts"]):
                        agentic_pass += 1
                print(f"[{vname}] agent:{t['id']:16} running...")
            results[vname] = {
                "codegen": {"pass": codegen_pass, "total": codegen_total,
                            "rate": codegen_pass / codegen_total},
                "agentic": {"pass": agentic_pass, "total": agentic_total,
                            "rate": agentic_pass / agentic_total,
                            "tool_used": tool_used, "tool_used_rate": tool_used / agentic_total},
            }
            r = results[vname]
            print(f"--> {vname}: codegen {codegen_pass}/{codegen_total} "
                  f"({r['codegen']['rate']:.1%}), agentic {agentic_pass}/{agentic_total} "
                  f"({r['agentic']['rate']:.1%}), tool-used {tool_used}/{agentic_total} "
                  f"({r['agentic']['tool_used_rate']:.1%})\n")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()

    out = Path(f"bench_hard_{model_file.split('.')[0][:20]}.json")
    out.write_text(json.dumps({"model": model_file, "k": k, "results": results}, indent=2))
    print("=== SUMMARY ===")
    print(f"model: {model_file}, k={k}")
    for v, r in results.items():
        print(f"  {v:9} codegen {r['codegen']['pass']}/{r['codegen']['total']} "
              f"({r['codegen']['rate']:.0%})  agentic {r['agentic']['pass']}/{r['agentic']['total']} "
              f"({r['agentic']['rate']:.0%})  tool-used {r['agentic']['tool_used_rate']:.0%}")
    print(f"raw: {out}")


if __name__ == "__main__":
    main()
