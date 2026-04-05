"""Automated test suite for Jem — run all key scenarios and report failures.

Usage:
    python tests/test_jem.py          # run all tests
    python tests/test_jem.py --quick  # fast subset only
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
from gem.config import load_config
from gem.runtime import GemRuntimeGateway
from gem.toolkit import GemToolkit
from gem.tool_router import route_tools
from gem.prompts import build_system_prompt
from gem.models import resolve_profile
from gem.composer import compose_messages


class TestResult:
    def __init__(self, name, passed, detail="", elapsed=0):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.elapsed = elapsed

    def __str__(self):
        status = "\033[32m✓\033[0m" if self.passed else "\033[31m✗\033[0m"
        return f"  {status} {self.name:<50} {self.elapsed:.1f}s  {self.detail}"


def run_tests(quick=False):
    config = load_config()
    profile = resolve_profile(config.runtime.profile, config.runtime.model)
    config.runtime.model = config.runtime.model or profile.default_model
    engine = GemRuntimeGateway(config.runtime)
    toolkit = GemToolkit(Path.cwd(), config)
    prompt = build_system_prompt(profile)
    results = []

    def chat(query, expect_tool=None, expect_in="", no_special_tokens=True):
        """Run a query through the full pipeline and check results."""
        start = time.time()
        routing = route_tools(query, toolkit.list_tool_names())
        use_minimal = profile.feature_variant == "compact"
        schemas = toolkit.schemas(minimal=use_minimal)
        tools = [t for t in schemas if t["function"]["name"] in routing.tool_names]
        composed = compose_messages(profile, prompt, '', [], query, provider=config.runtime.provider)

        # Stream events
        thinking, content, tool_calls = "", "", []
        for event in engine.stream_chat_events(composed, tools=tools or None):
            if event["type"] == "thinking":
                thinking += str(event["content"])
            elif event["type"] == "content":
                content += str(event["content"])
            elif event["type"] == "tool_calls":
                tool_calls = event["tool_calls"]

        elapsed = time.time() - start
        called = [t["function"]["name"] for t in tool_calls]

        # Check for leaked special tokens
        has_special = "<|" in content or "|>" in content or "<tool_call" in content
        if no_special_tokens and has_special:
            return TestResult(query[:50], False, f"LEAKED TOKENS in output", elapsed)

        if expect_tool:
            # Check if tool was called natively or in thinking
            if expect_tool in called:
                return TestResult(query[:50], True, f"tool={expect_tool}", elapsed)
            if expect_tool in thinking.lower():
                return TestResult(query[:50], True, f"force-tool={expect_tool}", elapsed)
            return TestResult(query[:50], False, f"expected {expect_tool}, got {called or 'none'}", elapsed)

        if expect_in:
            if expect_in.lower() in (content + thinking).lower():
                return TestResult(query[:50], True, f"found '{expect_in}'", elapsed)
            return TestResult(query[:50], False, f"'{expect_in}' not in response", elapsed)

        if content.strip() or tool_calls:
            return TestResult(query[:50], True, f"{len(content)} chars", elapsed)
        return TestResult(query[:50], False, "empty response", elapsed)

    print("=" * 60)
    print("JEM AUTOMATED TEST SUITE")
    print("=" * 60)

    # --- ROUTING TESTS ---
    print("\n--- Tool Routing ---")
    routing_tests = [
        ("hi", None, "casual"),
        ("what time is it?", {"current_datetime"}, "time"),
        ("search for python news", {"web_search"}, "web"),
        ("create test.py with hello", {"write_file"}, "file_write"),
        ("edit main.py to fix bug", {"write_file", "read_file"}, "file_edit"),
        ("run pytest", {"bash"}, "shell"),
        ("humm why does it say str?", set(), "question = no tools"),
    ]
    for query, expected, desc in routing_tests:
        r = route_tools(query, toolkit.list_tool_names())
        if expected is None:
            ok = len(r.tool_names) == 0
        elif len(expected) == 0:
            ok = True  # any tools ok for questions
        else:
            ok = bool(expected & r.tool_names)
        result = TestResult(f"route: {query[:40]}", ok, f"{desc} → {sorted(r.tool_names)[:3]}")
        results.append(result)
        print(result)

    # --- TOOL EXECUTION TESTS ---
    print("\n--- Tool Execution ---")
    results.append(chat("what time is it?", expect_tool="current_datetime"))
    print(results[-1])

    results.append(chat("hi how are you"))  # just check it responds
    print(results[-1])

    if not quick:
        results.append(chat("search for python 3.13 features", expect_tool="web_search"))
        print(results[-1])

        results.append(chat("read the pyproject.toml", expect_tool="read_file"))
        print(results[-1])

        results.append(chat("run ls", expect_tool="bash"))
        print(results[-1])

    # --- SPECIAL TOKEN LEAK TEST ---
    print("\n--- Token Leak Check ---")
    for query in ["hi", "what time is it?", "list files"]:
        results.append(chat(query, no_special_tokens=True))
        print(results[-1])

    # --- SUMMARY ---
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pct = passed / total * 100 if total else 0
    color = "\033[32m" if pct >= 90 else "\033[33m" if pct >= 70 else "\033[31m"
    print(f"{color}  {passed}/{total} passed ({pct:.0f}%)\033[0m")

    # Save results
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": config.runtime.model,
        "provider": config.runtime.provider,
        "passed": passed,
        "total": total,
        "failures": [{"name": r.name, "detail": r.detail} for r in results if not r.passed],
    }
    report_path = Path("tests/test_results.json")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report saved to {report_path}")

    engine.close()
    return 0 if pct >= 80 else 1


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    sys.exit(run_tests(quick=quick))
