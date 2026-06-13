#!/usr/bin/env python3
"""
Benchmark script comparing BF16 and Q4_K_M quantizations of DiffusionGemma.

Runs ~8 small Python coding tasks, sampling each k=3 times per quantization.
Measures pass@1, generation time, and throughput.
"""

import os
import sys
import time
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure PYTHONPATH includes src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from localcode.config import RuntimeConfig
from localcode.runtime import LocalCodeRuntimeGateway


# ============================================================================
# Task Definitions (HumanEval-style)
# ============================================================================

TASKS = [
    {
        "id": "palindrome",
        "prompt": "Write a Python function `is_palindrome(s)` that returns True if s reads the same forwards and backwards, ignoring case and spaces.",
        "asserts": [
            ('is_palindrome("Race car") == True', "Race car"),
            ('is_palindrome("hello") == False', "hello"),
            ('is_palindrome("A man a plan a canal Panama") == True', "A man a plan a canal Panama"),
            ('is_palindrome("") == True', "empty string"),
        ],
    },
    {
        "id": "sum_list",
        "prompt": "Write a Python function `sum_list(lst)` that returns the sum of all numbers in the list.",
        "asserts": [
            ("sum_list([1, 2, 3]) == 6", "basic"),
            ("sum_list([]) == 0", "empty"),
            ("sum_list([-1, -2, -3]) == -6", "negative"),
            ("sum_list([1.5, 2.5, 1.0]) == 5.0", "floats"),
        ],
    },
    {
        "id": "is_prime",
        "prompt": "Write a Python function `is_prime(n)` that returns True if n is a prime number.",
        "asserts": [
            ("is_prime(2) == True", "2"),
            ("is_prime(17) == True", "17"),
            ("is_prime(1) == False", "1"),
            ("is_prime(18) == False", "18"),
            ("is_prime(0) == False", "0"),
        ],
    },
    {
        "id": "reverse_string",
        "prompt": "Write a Python function `reverse_string(s)` that returns the string reversed.",
        "asserts": [
            ('reverse_string("hello") == "olleh"', "hello"),
            ('reverse_string("") == ""', "empty"),
            ('reverse_string("a") == "a"', "single char"),
            ('reverse_string("Python") == "nohtyP"', "Python"),
        ],
    },
    {
        "id": "max_element",
        "prompt": "Write a Python function `max_element(lst)` that returns the maximum element in a list.",
        "asserts": [
            ("max_element([1, 5, 3]) == 5", "basic"),
            ("max_element([-1, -5, -3]) == -1", "negative"),
            ("max_element([42]) == 42", "single"),
            ("max_element([10, 10, 10]) == 10", "duplicates"),
        ],
    },
    {
        "id": "fizzbuzz",
        "prompt": "Write a Python function `fizzbuzz(n)` that returns a list where: numbers divisible by 3 are 'Fizz', by 5 are 'Buzz', by both are 'FizzBuzz', otherwise the number itself.",
        "asserts": [
            ('fizzbuzz(15) == [1, 2, "Fizz", 4, "Buzz", "Fizz", 7, 8, "Fizz", "Buzz", 11, "Fizz", 13, 14, "FizzBuzz"]', "basic"),
            ('fizzbuzz(3) == [1, 2, "Fizz"]', "short"),
        ],
    },
    {
        "id": "count_vowels",
        "prompt": "Write a Python function `count_vowels(s)` that returns the number of vowels in a string (a, e, i, o, u, case-insensitive).",
        "asserts": [
            ('count_vowels("hello") == 2', "hello"),
            ('count_vowels("AEIOU") == 5', "all vowels"),
            ('count_vowels("bcdfg") == 0', "no vowels"),
            ('count_vowels("") == 0', "empty"),
        ],
    },
    {
        "id": "factorial",
        "prompt": "Write a Python function `factorial(n)` that returns the factorial of n.",
        "asserts": [
            ("factorial(0) == 1", "0"),
            ("factorial(5) == 120", "5"),
            ("factorial(1) == 1", "1"),
            ("factorial(10) == 3628800", "10"),
        ],
    },
]


# ============================================================================
# Utility Functions
# ============================================================================

def extract_python_code(text: str) -> str:
    """
    Extract Python code from model output. Strips markdown fences and prose.
    """
    # Remove markdown fences
    text = re.sub(r"```python\n", "", text)
    text = re.sub(r"```\n", "", text)
    text = re.sub(r"```", "", text)

    # Try to extract just the function definition(s)
    # Look for lines starting with 'def '
    lines = text.split("\n")
    code_lines = []
    in_function = False
    indent_level = 0

    for line in lines:
        if line.strip().startswith("def "):
            in_function = True
            indent_level = len(line) - len(line.lstrip())
            code_lines.append(line)
        elif in_function:
            if line.strip() == "":
                code_lines.append(line)
            elif len(line) - len(line.lstrip()) > indent_level or line.strip() == "":
                code_lines.append(line)
            elif line.strip() and not line.startswith(" " * (indent_level + 1)):
                # End of function
                break
            else:
                code_lines.append(line)

    code = "\n".join(code_lines).strip()
    if not code:
        code = text.strip()
    return code


def run_task_sample(
    gw: LocalCodeRuntimeGateway, task: Dict[str, Any], timeout: float = 60.0
) -> Tuple[bool, float, str]:
    """
    Run a single sample of a task. Returns (passed, generation_time_seconds, code).
    """
    prompt = task["prompt"]
    asserts = task["asserts"]

    system_msg = {
        "role": "system",
        "content": "You are LocalCode, a coding assistant. Reply with ONLY the requested Python code, no prose.",
    }
    user_msg = {"role": "user", "content": prompt}

    # Stream and collect output
    start_time = time.time()
    content = []
    try:
        for ev in gw.stream_chat_events(
            [system_msg, user_msg], tools=None, num_predict=512
        ):
            if ev["type"] == "content":
                content.append(ev["content"])
    except Exception as e:
        print(f"    ERROR during generation: {e}")
        return False, time.time() - start_time, ""

    gen_time = time.time() - start_time

    if gen_time > timeout:
        print(f"    TIMEOUT: generation took {gen_time:.1f}s > {timeout}s")
        return False, gen_time, ""

    code = "".join(content)
    code = extract_python_code(code)

    # Try to execute and run asserts
    try:
        namespace = {}
        exec(code, namespace)

        # Run all asserts
        for assert_expr, desc in asserts:
            try:
                result = eval(assert_expr, namespace)
                if not result:
                    print(f"    FAIL: {desc} ({assert_expr})")
                    return False, gen_time, code
            except Exception as e:
                print(f"    FAIL: {desc} raised {type(e).__name__}: {e}")
                return False, gen_time, code

        # All asserts passed
        return True, gen_time, code

    except Exception as e:
        print(f"    FAIL: exec raised {type(e).__name__}: {e}")
        return False, gen_time, code


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    # Model paths
    bf16_model = os.path.expanduser(
        "~/.local/share/localcode/models/diffusiongemma-26B-A4B-it-BF16.gguf"
    )
    q4_model = os.path.expanduser(
        "~/.local/share/localcode/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
    )

    # Verify models exist
    for model_path in [bf16_model, q4_model]:
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found: {model_path}")
            sys.exit(1)

    print("=" * 80)
    print("DiffusionGemma Quantization Benchmark")
    print("=" * 80)
    print(f"Tasks: {len(TASKS)}")
    print(f"Samples per task/quant: 3")
    print(f"BF16 Model: {bf16_model}")
    print(f"Q4 Model: {q4_model}")
    print("=" * 80)
    print()

    results = {"BF16": {}, "Q4": {}}
    quants = [
        ("BF16", bf16_model),
        ("Q4", q4_model),
    ]

    for quant_name, model_path in quants:
        print(f"\n{'='*80}")
        print(f"QUANTIZATION: {quant_name}")
        print(f"{'='*80}")

        cfg = RuntimeConfig()
        cfg.provider = "llama_cpp"
        cfg.model = model_path

        print(f"Loading gateway for {model_path}...")
        try:
            gw = LocalCodeRuntimeGateway(cfg)
        except Exception as e:
            print(f"ERROR: Failed to load gateway: {e}")
            sys.exit(1)

        results[quant_name] = {
            "tasks": {},
            "total_passes": 0,
            "total_samples": 0,
            "gen_times": [],
        }

        # Run each task 3 times
        for task in TASKS:
            task_id = task["id"]
            print(f"\n  Task: {task_id}")

            results[quant_name]["tasks"][task_id] = {
                "passes": 0,
                "samples": 3,
                "gen_times": [],
                "codes": [],
            }

            for sample_idx in range(3):
                print(f"    Sample {sample_idx + 1}/3...", end=" ", flush=True)
                passed, gen_time, code = run_task_sample(gw, task)

                results[quant_name]["tasks"][task_id]["gen_times"].append(gen_time)
                results[quant_name]["tasks"][task_id]["codes"].append(code)

                if passed:
                    results[quant_name]["tasks"][task_id]["passes"] += 1
                    results[quant_name]["total_passes"] += 1
                    print(f"PASS ({gen_time:.2f}s)")
                else:
                    print(f"FAIL ({gen_time:.2f}s)")

                results[quant_name]["total_samples"] += 1

            # Per-task pass rate
            pass_rate = results[quant_name]["tasks"][task_id]["passes"] / 3
            avg_time = sum(results[quant_name]["tasks"][task_id]["gen_times"]) / 3
            print(f"    -> Pass rate: {pass_rate:.1%}, Avg time: {avg_time:.2f}s")

        # Delete gateway to free memory
        del gw

    # ========================================================================
    # Report
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for quant_name in ["BF16", "Q4"]:
        res = results[quant_name]
        total_passes = res["total_passes"]
        total_samples = res["total_samples"]
        pass_at_1 = total_passes / total_samples if total_samples > 0 else 0

        # Collect all generation times
        all_gen_times = []
        for task_res in res["tasks"].values():
            all_gen_times.extend(task_res["gen_times"])

        mean_gen_time = sum(all_gen_times) / len(all_gen_times) if all_gen_times else 0

        # Estimate tokens per second
        # Assume ~4 chars per token on average
        total_tokens = sum(
            len(code) // 4 for task_res in res["tasks"].values() for code in task_res["codes"]
        )
        total_gen_time = sum(all_gen_times)
        tok_per_sec = (
            total_tokens / total_gen_time if total_gen_time > 0 else 0
        )

        print(f"\n{quant_name}:")
        print(f"  Pass@1:         {pass_at_1:.1%} ({total_passes}/{total_samples})")
        print(f"  Mean gen time:  {mean_gen_time:.2f}s")
        print(f"  Est. tok/sec:   {tok_per_sec:.1f}")
        print(f"\n  Per-task breakdown:")
        for task_id, task_res in res["tasks"].items():
            pass_rate = task_res["passes"] / task_res["samples"]
            avg_time = sum(task_res["gen_times"]) / len(task_res["gen_times"])
            print(
                f"    {task_id:20} {pass_rate:.1%} ({task_res['passes']}/{task_res['samples']}) - {avg_time:.2f}s"
            )

    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    bf16_pass = results["BF16"]["total_passes"] / results["BF16"]["total_samples"]
    q4_pass = results["Q4"]["total_passes"] / results["Q4"]["total_samples"]

    bf16_times = []
    q4_times = []
    for task_res in results["BF16"]["tasks"].values():
        bf16_times.extend(task_res["gen_times"])
    for task_res in results["Q4"]["tasks"].values():
        q4_times.extend(task_res["gen_times"])

    bf16_mean_time = sum(bf16_times) / len(bf16_times)
    q4_mean_time = sum(q4_times) / len(q4_times)

    print(f"\nQuant   | Pass@1  | Mean Time | Ratio")
    print(f"--------|---------|-----------|-------")
    print(f"BF16    | {bf16_pass:.1%}   | {bf16_mean_time:6.2f}s  | 1.0x")
    print(f"Q4      | {q4_pass:.1%}   | {q4_mean_time:6.2f}s  | {q4_mean_time/bf16_mean_time:.2f}x")

    # Summary
    print("\n" + "-" * 80)
    pass_diff_pct = (bf16_pass - q4_pass) * 100
    time_diff_pct = (q4_mean_time - bf16_mean_time) / bf16_mean_time * 100

    print("Summary:")
    if abs(pass_diff_pct) < 5:
        print(
            f"  BF16 and Q4 have similar pass rates (diff: {pass_diff_pct:+.1f}%)."
        )
    else:
        winner = "BF16" if pass_diff_pct > 0 else "Q4"
        print(
            f"  {winner} has significantly better accuracy (diff: {abs(pass_diff_pct):.1f}%)."
        )

    if time_diff_pct > 10:
        print(f"  Q4 is slower (by {time_diff_pct:+.1f}%).")
    elif time_diff_pct < -10:
        print(f"  Q4 is faster (by {abs(time_diff_pct):.1f}%).")
    else:
        print(f"  Generation speed is similar (diff: {time_diff_pct:+.1f}%).")

    # Save raw results
    results_json = repo_root / "bench_diffusion_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to: {results_json}")


if __name__ == "__main__":
    main()
