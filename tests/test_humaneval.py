"""HumanEval mini-benchmark — 5 representative problems.

Tests the model's code generation ability by feeding function signatures
with docstrings and checking if generated completions pass unit tests.

Usage:
    python tests/test_humaneval.py
    python tests/test_humaneval.py --verbose
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── 5 HumanEval Problems ────────────────────────────────────────────────

PROBLEMS = [
    {
        "id": "HumanEval/0",
        "name": "has_close_elements",
        "difficulty": "easy",
        "prompt": textwrap.dedent("""\
            from typing import List

            def has_close_elements(numbers: List[float], threshold: float) -> bool:
                \"\"\"Check if in given list of numbers, are any two numbers closer to each other than given threshold.
                >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
                False
                >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.2], 0.3)
                True
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
            assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
            assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
            assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0], 2.0) == True
            assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
            assert has_close_elements([1.0, 2.0, 3.1, 4.0, 5.0], 0.5) == False
        """),
    },
    {
        "id": "HumanEval/8",
        "name": "sum_product",
        "difficulty": "easy-medium",
        "prompt": textwrap.dedent("""\
            from typing import List, Tuple

            def sum_product(numbers: List[int]) -> Tuple[int, int]:
                \"\"\"For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
                Empty sum should be equal to 0 and empty product should be equal to 1.
                >>> sum_product([])
                (0, 1)
                >>> sum_product([1, 2, 3, 4])
                (10, 24)
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert sum_product([]) == (0, 1)
            assert sum_product([1, 1, 1]) == (3, 1)
            assert sum_product([100, 0]) == (100, 0)
            assert sum_product([3, 5, 7]) == (15, 105)
            assert sum_product([10]) == (10, 10)
        """),
    },
    {
        "id": "HumanEval/48",
        "name": "is_palindrome",
        "difficulty": "medium",
        "prompt": textwrap.dedent("""\
            def is_palindrome(text: str) -> bool:
                \"\"\"Checks if given string is a palindrome.
                >>> is_palindrome('')
                True
                >>> is_palindrome('aba')
                True
                >>> is_palindrome('zbcd')
                False
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert is_palindrome('') == True
            assert is_palindrome('aba') == True
            assert is_palindrome('aaaaa') == True
            assert is_palindrome('zbcd') == False
            assert is_palindrome('xywyx') == True
            assert is_palindrome('a') == True
            assert is_palindrome('ab') == False
        """),
    },
    {
        "id": "HumanEval/5",
        "name": "intersperse",
        "difficulty": "medium",
        "prompt": textwrap.dedent("""\
            from typing import List

            def intersperse(numbers: List[int], delimeter: int) -> List[int]:
                \"\"\"Insert a number 'delimeter' between every two consecutive elements of input list `numbers`.
                >>> intersperse([], 4)
                []
                >>> intersperse([1, 2, 3], 4)
                [1, 4, 2, 4, 3]
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert intersperse([], 4) == []
            assert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]
            assert intersperse([1], 4) == [1]
            assert intersperse([1, 2], 4) == [1, 4, 2]
            assert intersperse([1, 2, 3, 4, 5], 0) == [1, 0, 2, 0, 3, 0, 4, 0, 5]
        """),
    },
    {
        "id": "HumanEval/3",
        "name": "separate_paren_groups",
        "difficulty": "hard",
        "prompt": textwrap.dedent("""\
            from typing import List

            def separate_paren_groups(paren_string: str) -> List[str]:
                \"\"\"Input to this function is a string containing multiple groups of nested parentheses.
                Your goal is to separate those groups into separate strings and return the list of those.
                Separate groups are balanced (each open brace is properly closed) and not nested within each other.
                Ignore any spaces in the input string.
                >>> separate_paren_groups('( ) (( )) (( )( ))')
                ['()', '(())', '(()())']
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
            assert separate_paren_groups('() (()) ((()))') == ['()', '(())', '((()))']
            assert separate_paren_groups('(()(()))') == ['(()(()))']
        """),
    },
    # ── Batch 2: problems 6-15 ──
    {
        "id": "HumanEval/2",
        "name": "truncate_number",
        "difficulty": "easy",
        "prompt": textwrap.dedent("""\
            def truncate_number(number: float) -> float:
                \"\"\"Given a positive floating point number, it can be decomposed into
                an integer part (largest integer smaller than given number) and decimals
                (leftover part always smaller than 1).
                Return the decimal part of the number.
                >>> truncate_number(3.5)
                0.5
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert truncate_number(3.5) == 0.5
            assert abs(truncate_number(1.25) - 0.25) < 1e-6
            assert abs(truncate_number(123.0) - 0.0) < 1e-6
        """),
    },
    {
        "id": "HumanEval/4",
        "name": "mean_absolute_deviation",
        "difficulty": "easy-medium",
        "prompt": textwrap.dedent("""\
            from typing import List

            def mean_absolute_deviation(numbers: List[float]) -> float:
                \"\"\"For a given list of input numbers, calculate Mean Absolute Deviation
                around the mean of this dataset.
                Mean Absolute Deviation is the average absolute difference between each
                element and a centerpoint (mean in this case):
                MAD = average | x - x_mean |
                >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
                1.0
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
            assert abs(mean_absolute_deviation([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6
            assert abs(mean_absolute_deviation([1.0]) - 0.0) < 1e-6
        """),
    },
    {
        "id": "HumanEval/6",
        "name": "parse_nested_parens",
        "difficulty": "medium",
        "prompt": textwrap.dedent("""\
            from typing import List

            def parse_nested_parens(paren_string: str) -> List[int]:
                \"\"\"Input to this function is a string represented multiple groups of nested parentheses separated by spaces.
                For each of the group, output the deepest level of nesting of parentheses.
                E.g. (()()) has maximum two levels of nesting while ((())) has three.
                >>> parse_nested_parens('(()()) ((())) () ((())()())')
                [2, 3, 1, 3]
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert parse_nested_parens('(()()) ((())) () ((())()())') == [2, 3, 1, 3]
            assert parse_nested_parens('() (()) ((())) (((())))') == [1, 2, 3, 4]
            assert parse_nested_parens('(()(())(()))') == [3]
        """),
    },
    {
        "id": "HumanEval/7",
        "name": "filter_by_substring",
        "difficulty": "easy",
        "prompt": textwrap.dedent("""\
            from typing import List

            def filter_by_substring(strings: List[str], substring: str) -> List[str]:
                \"\"\"Filter an input list of strings only for ones that contain given substring.
                >>> filter_by_substring([], 'a')
                []
                >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
                ['abc', 'bacd', 'array']
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert filter_by_substring([], 'a') == []
            assert filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a') == ['abc', 'bacd', 'array']
            assert filter_by_substring(['xxx', 'asd', 'xxy', 'john doe', 'xxxAAA', 'xxx'], 'xxx') == ['xxx', 'xxxAAA', 'xxx']
        """),
    },
    {
        "id": "HumanEval/9",
        "name": "rolling_max",
        "difficulty": "medium",
        "prompt": textwrap.dedent("""\
            from typing import List

            def rolling_max(numbers: List[int]) -> List[int]:
                \"\"\"From a given list of integers, generate a list of rolling maximum element found until given moment in the sequence.
                >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
                [1, 2, 3, 3, 3, 4, 4]
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert rolling_max([]) == []
            assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]
            assert rolling_max([4, 3, 2, 1]) == [4, 4, 4, 4]
            assert rolling_max([1]) == [1]
        """),
    },
    {
        "id": "HumanEval/11",
        "name": "string_xor",
        "difficulty": "easy-medium",
        "prompt": textwrap.dedent("""\
            def string_xor(a: str, b: str) -> str:
                \"\"\"Input are two strings a and b consisting only of 1s and 0s.
                Perform binary XOR on these inputs and return result also as a string.
                >>> string_xor('010', '110')
                '100'
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert string_xor('111000', '101010') == '010010'
            assert string_xor('1', '1') == '0'
            assert string_xor('0101', '0000') == '0101'
        """),
    },
    {
        "id": "HumanEval/12",
        "name": "longest",
        "difficulty": "easy",
        "prompt": textwrap.dedent("""\
            from typing import List, Optional

            def longest(strings: List[str]) -> Optional[str]:
                \"\"\"Out of list of strings, return the longest one. Return the first one in case of multiple
                strings of the same length. Return None in case the input list is empty.
                >>> longest([])

                >>> longest(['a', 'b', 'c'])
                'a'
                >>> longest(['a', 'bb', 'ccc'])
                'ccc'
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert longest([]) == None
            assert longest(['x', 'y', 'z']) == 'x'
            assert longest(['x', 'yyy', 'zzzz', 'www', 'kkkk', 'abc']) == 'zzzz'
        """),
    },
    {
        "id": "HumanEval/13",
        "name": "greatest_common_divisor",
        "difficulty": "easy",
        "prompt": textwrap.dedent("""\
            def greatest_common_divisor(a: int, b: int) -> int:
                \"\"\"Return a greatest common divisor of two integers a and b.
                >>> greatest_common_divisor(3, 5)
                1
                >>> greatest_common_divisor(25, 15)
                5
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert greatest_common_divisor(3, 7) == 1
            assert greatest_common_divisor(10, 15) == 5
            assert greatest_common_divisor(49, 14) == 7
            assert greatest_common_divisor(144, 60) == 12
        """),
    },
    {
        "id": "HumanEval/15",
        "name": "string_sequence",
        "difficulty": "easy",
        "prompt": textwrap.dedent("""\
            def string_sequence(n: int) -> str:
                \"\"\"Return a string containing space-delimited numbers starting from 0 upto n inclusive.
                >>> string_sequence(0)
                '0'
                >>> string_sequence(5)
                '0 1 2 3 4 5'
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert string_sequence(0) == '0'
            assert string_sequence(5) == '0 1 2 3 4 5'
            assert string_sequence(3) == '0 1 2 3'
        """),
    },
    {
        "id": "HumanEval/18",
        "name": "how_many_times",
        "difficulty": "medium",
        "prompt": textwrap.dedent("""\
            def how_many_times(string: str, substring: str) -> int:
                \"\"\"Find how many times a given substring can be found in the original string. Count overlapping cases.
                >>> how_many_times('', 'a')
                0
                >>> how_many_times('aaa', 'a')
                3
                >>> how_many_times('aaaa', 'aa')
                3
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert how_many_times('', 'x') == 0
            assert how_many_times('aaaa', 'aa') == 3
            assert how_many_times('aaa', 'a') == 3
            assert how_many_times('abcabc', 'abc') == 2
        """),
    },
    # ── Batch 3: problems 16-25 ──
    {
        "id": "HumanEval/20",
        "name": "find_closest_elements",
        "difficulty": "medium",
        "prompt": textwrap.dedent("""\
            from typing import List, Tuple

            def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
                \"\"\"From a supplied list of numbers (of length at least two) select and return two that are the closest to each
                other and return them in order (smaller number, larger number).
                >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])
                (2.0, 2.2)
                >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0])
                (2.0, 2.0)
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) == (2.0, 2.2)
            assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) == (2.0, 2.0)
            assert find_closest_elements([1.0, 5.0, 2.0]) == (1.0, 2.0)
        """),
    },
    {
        "id": "HumanEval/22",
        "name": "filter_integers",
        "difficulty": "easy",
        "prompt": textwrap.dedent("""\
            from typing import List, Any

            def filter_integers(values: List[Any]) -> List[int]:
                \"\"\"Filter given list of any python values only for integers.
                >>> filter_integers(['a', 3.14, 5])
                [5]
                >>> filter_integers([1, 2, 3, 'abc', {}, []])
                [1, 2, 3]
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert filter_integers([]) == []
            assert filter_integers([4, {}, [], 23.2, 9, 'adasd']) == [4, 9]
            assert filter_integers([3, 'c', 3, 3, 'a', 'b']) == [3, 3, 3]
        """),
    },
    {
        "id": "HumanEval/23",
        "name": "strlen",
        "difficulty": "easy",
        "prompt": textwrap.dedent("""\
            def strlen(string: str) -> int:
                \"\"\"Return length of given string.
                >>> strlen('')
                0
                >>> strlen('abc')
                3
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert strlen('') == 0
            assert strlen('x') == 1
            assert strlen('asdasnakj') == 9
        """),
    },
    {
        "id": "HumanEval/25",
        "name": "factorize",
        "difficulty": "medium",
        "prompt": textwrap.dedent("""\
            from typing import List

            def factorize(n: int) -> List[int]:
                \"\"\"Return list of prime factors of given integer in the order from smallest to largest.
                Each of the factors should be listed number of times corresponding to how many times it appears in factorization.
                Input number should be equal to the product of all factors.
                >>> factorize(8)
                [2, 2, 2]
                >>> factorize(25)
                [5, 5]
                >>> factorize(70)
                [2, 5, 7]
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert factorize(2) == [2]
            assert factorize(8) == [2, 2, 2]
            assert factorize(25) == [5, 5]
            assert factorize(70) == [2, 5, 7]
            assert factorize(3 * 5 * 7 * 11) == [3, 5, 7, 11]
        """),
    },
    {
        "id": "HumanEval/26",
        "name": "remove_duplicates",
        "difficulty": "easy-medium",
        "prompt": textwrap.dedent("""\
            from typing import List

            def remove_duplicates(numbers: List[int]) -> List[int]:
                \"\"\"From a list of integers, remove all elements that occur more than once.
                Keep order of elements left the same as in the input.
                >>> remove_duplicates([1, 2, 3, 2, 4])
                [1, 3, 4]
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert remove_duplicates([]) == []
            assert remove_duplicates([1, 2, 3, 2, 4]) == [1, 3, 4]
            assert remove_duplicates([1, 2, 3, 2, 4, 3, 5]) == [1, 5]
        """),
    },
    {
        "id": "HumanEval/28",
        "name": "concatenate",
        "difficulty": "easy",
        "prompt": textwrap.dedent("""\
            from typing import List

            def concatenate(strings: List[str]) -> str:
                \"\"\"Concatenate list of strings into a single string.
                >>> concatenate([])
                ''
                >>> concatenate(['a', 'b', 'c'])
                'abc'
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert concatenate([]) == ''
            assert concatenate(['x', 'y', 'z']) == 'xyz'
            assert concatenate(['x', 'y', 'z', 'w', 'k']) == 'xyzwk'
        """),
    },
    {
        "id": "HumanEval/29",
        "name": "filter_by_prefix",
        "difficulty": "easy",
        "prompt": textwrap.dedent("""\
            from typing import List

            def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
                \"\"\"Filter an input list of strings only for ones that start with a given prefix.
                >>> filter_by_prefix([], 'a')
                []
                >>> filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a')
                ['abc', 'array']
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert filter_by_prefix([], 'a') == []
            assert filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a') == ['abc', 'array']
            assert filter_by_prefix(['abc', 'azy'], 'a') == ['abc', 'azy']
        """),
    },
    {
        "id": "HumanEval/30",
        "name": "get_positive",
        "difficulty": "easy",
        "prompt": textwrap.dedent("""\
            from typing import List

            def get_positive(l: List[int]) -> List[int]:
                \"\"\"Return only positive numbers in the list.
                >>> get_positive([-1, 2, -4, 5, 6])
                [2, 5, 6]
                >>> get_positive([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
                [5, 3, 2, 3, 9, 123, 1]
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert get_positive([-1, -2, 4, 5, 6]) == [4, 5, 6]
            assert get_positive([5, 3, -5, 2, 3, 9, 0, 123, 1, -10]) == [5, 3, 2, 3, 9, 123, 1]
            assert get_positive([-1, -2]) == []
        """),
    },
    {
        "id": "HumanEval/32",
        "name": "find_zero",
        "difficulty": "hard",
        "prompt": textwrap.dedent("""\
            import math

            def poly(xs: list, x: float) -> float:
                \"\"\"Evaluates polynomial with coefficients xs at point x.
                return xs[0] + xs[1] * x + xs[2] * x^2 + ....
                \"\"\"
                return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])

            def find_zero(xs: list) -> float:
                \"\"\"xs are coefficients of a polynomial. find_zero finds x such that poly(xs, x) = 0.
                find_zero returns only one zero point, even if there are many.
                Moreover, find_zero only takes list xs having even number of coefficients
                and largest non zero coefficient as it guarantees a solution.
                >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x
                -0.5
                >>> round(find_zero([-6, 11, -6, 1]), 2) # (x - 1)(x - 2)(x - 3) = -6 + 11x - 6x^2 + x^3
                1.0
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert abs(poly([1, 2], find_zero([1, 2]))) < 1e-4
            assert abs(poly([-6, 11, -6, 1], find_zero([-6, 11, -6, 1]))) < 1e-4
            assert abs(poly([2, -3, 1], find_zero([2, -3, 1]))) < 1e-4
        """),
    },
    {
        "id": "HumanEval/34",
        "name": "unique",
        "difficulty": "easy",
        "prompt": textwrap.dedent("""\
            from typing import List

            def unique(l: List[int]) -> List[int]:
                \"\"\"Return sorted unique elements in a list.
                >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])
                [0, 2, 3, 5, 9, 123]
                \"\"\"
        """),
        "tests": textwrap.dedent("""\
            assert unique([5, 3, 5, 2, 3, 3, 9, 0, 123]) == [0, 2, 3, 5, 9, 123]
            assert unique([]) == []
            assert unique([1]) == [1]
        """),
    },
]


def generate_completion(engine, prompt: str) -> str:
    """Feed prompt to model, get completion with thinking disabled."""
    messages = [
        {"role": "system", "content": "Complete the Python function. Return ONLY the function body code, properly indented with 4 spaces. No explanation, no markdown fences."},
        {"role": "user", "content": prompt},
    ]
    # Use stream_chat_events with think=False to skip thinking overhead
    parts = []
    for event in engine.stream_chat_events(messages, think=False):
        if event["type"] == "content":
            parts.append(str(event["content"]))
    result = "".join(parts)
    # Strip any leaked thinking tokens
    import re
    result = result.replace("<unused25>", "")
    result = re.sub(r"<\|channel>thought\n?", "", result)
    result = re.sub(r"<channel\|>\n?", "", result)
    return result


def extract_function_body(completion: str, prompt: str) -> str:
    """Extract just the function body from the completion."""
    import re
    # Strip markdown fences
    completion = re.sub(r"```python\s*\n?", "", completion)
    completion = re.sub(r"```\s*\n?", "", completion)
    completion = completion.rstrip()  # only strip trailing, preserve leading indent

    # If completion includes the full function def, use it standalone
    if "def " in completion.lstrip()[:10]:
        return completion

    # Dedent to zero, then re-indent to 4 spaces
    lines = completion.split("\n")
    # Find minimum indentation (ignoring empty lines)
    min_indent = 999
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            min_indent = min(min_indent, indent)
    if min_indent == 999:
        min_indent = 0

    # Re-indent all lines to exactly 4 spaces
    normalized = []
    for line in lines:
        if line.strip():
            dedented = line[min_indent:] if len(line) >= min_indent else line.lstrip()
            normalized.append(f"    {dedented}")
        else:
            normalized.append("")

    # Ensure prompt ends with newline
    p = prompt.rstrip() + "\n"
    return p + "\n".join(normalized)


def _fix_indentation(code: str) -> str:
    """Try to fix inconsistent indentation from quantized model output.

    The IQ3_S quant sometimes mixes indent levels within a function.
    This attempts to normalize by detecting the intended structure.
    """
    try:
        compile(code, "<test>", "exec")
        return code  # already valid
    except IndentationError:
        pass

    # Try autopep8 if available
    try:
        import autopep8
        fixed = autopep8.fix_code(code, options={"max_line_length": 120})
        compile(fixed, "<test>", "exec")
        return fixed
    except Exception:
        pass

    # Manual fix: find the function body and normalize indent levels
    lines = code.split("\n")
    fixed_lines = []
    in_body = False
    indent_stack = [0]

    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            fixed_lines.append("")
            continue

        if stripped.startswith("def ") and not in_body:
            fixed_lines.append(stripped)  # top-level def at col 0
            in_body = True
            indent_stack = [0, 4]  # function body at 4
            continue

        if not in_body:
            fixed_lines.append(line)
            continue

        # We're in the function body — normalize based on structure
        current_indent = len(line) - len(stripped)

        # Dedent signals (return, break, continue, else, elif, except, finally)
        if stripped.startswith(("return ", "return\n", "break", "continue")):
            # Should be at current block level
            level = indent_stack[-1] if indent_stack else 4
            fixed_lines.append(" " * level + stripped)
        elif stripped.startswith(("else:", "elif ", "except", "finally:", "except:")):
            # Same level as the if/try
            if len(indent_stack) > 2:
                level = indent_stack[-2]
            else:
                level = indent_stack[-1] if indent_stack else 4
            fixed_lines.append(" " * level + stripped)
            if stripped.endswith(":"):
                if len(indent_stack) > 1 and indent_stack[-1] > level:
                    indent_stack[-1] = level + 4
                else:
                    indent_stack.append(level + 4)
        elif stripped.endswith(":"):
            # Block opener — use current level, push next level
            level = indent_stack[-1] if indent_stack else 4
            fixed_lines.append(" " * level + stripped)
            indent_stack.append(level + 4)
        else:
            # Regular statement — use current deepest level
            level = indent_stack[-1] if indent_stack else 4
            fixed_lines.append(" " * level + stripped)

    result = "\n".join(fixed_lines)
    try:
        compile(result, "<test>", "exec")
        return result
    except Exception:
        return code  # give up, return original


def run_test(code: str, tests: str, timeout: int = 30) -> tuple[bool, str]:
    """Execute code + tests in a subprocess. Returns (passed, error_msg)."""
    # Try to fix indentation issues from quantized model
    code = _fix_indentation(code)
    full_code = code + "\n\n" + tests + "\nprint('ALL_TESTS_PASSED')\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=timeout,
            )
            if "ALL_TESTS_PASSED" in result.stdout:
                return True, ""
            error = (result.stderr or result.stdout).strip()
            return False, error[:300]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        finally:
            Path(f.name).unlink(missing_ok=True)


def run_benchmark(verbose: bool = False) -> None:
    """Run 5 HumanEval problems against the local model."""
    from gem.config import load_config
    from gem.runtime import GemRuntimeGateway

    config = load_config()
    engine = GemRuntimeGateway(config.runtime)

    print(f"\n{'='*60}")
    print(f"  HumanEval Benchmark ({len(PROBLEMS)} problems, pass@2)")
    print(f"  Model: {config.runtime.model}")
    print(f"{'='*60}\n")

    passed = 0
    total = len(PROBLEMS)
    results = []
    MAX_ATTEMPTS = 2  # pass@2: try twice per problem

    for i, problem in enumerate(PROBLEMS, 1):
        name = problem["name"]
        difficulty = problem["difficulty"]
        print(f"  [{i}/{total}] {name} ({difficulty})...", end=" ", flush=True)

        t0 = time.time()
        problem_passed = False
        last_error = ""
        last_completion = ""

        for attempt in range(MAX_ATTEMPTS):
            try:
                completion = generate_completion(engine, problem["prompt"])
                code = extract_function_body(completion, problem["prompt"])
                ok, error = run_test(code, problem["tests"])

                if ok:
                    problem_passed = True
                    break
                last_error = error
                last_completion = completion
            except Exception as e:
                last_error = str(e)

        elapsed = time.time() - t0
        if problem_passed:
            passed += 1
            attempts_str = f" (attempt {attempt+1})" if attempt > 0 else ""
            print(f"✓ PASS ({elapsed:.1f}s){attempts_str}")
            results.append(("PASS", elapsed))
        else:
            print(f"✗ FAIL ({elapsed:.1f}s)")
            results.append(("FAIL", elapsed))
            if verbose:
                print(f"    Error: {last_error[:200]}")
                print(f"    Completion:\n{textwrap.indent(last_completion[:300], '      ')}")

    # Summary
    total_time = sum(r[1] for r in results)
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed ({passed/total*100:.0f}%)")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg per problem: {total_time/total:.1f}s")
    print(f"{'='*60}\n")

    # Return exit code
    return 0 if passed == total else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HumanEval mini-benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show errors and completions")
    args = parser.parse_args()
    sys.exit(run_benchmark(verbose=args.verbose))
