"""Parse tool calls from Gemma 4 model output using its native format.

Gemma 4 uses a specific special-token format for function calling:

Tool declarations (in system prompt):
  <|tool>declaration:FUNC_NAME{description:<|"|>DESC<|"|>,parameters:{properties:{
    PARAM:{description:<|"|>DESC<|"|>,type:<|"|>TYPE<|"|>}},required:[<|"|>PARAM<|"|>],
    type:<|"|>object<|"|>}}<tool|>

Function calls (model output):
  <|tool_call>call:FUNC_NAME{param:<|"|>value<|"|>,param2:<|"|>value2<|"|>}<tool_call|>

Tool responses (fed back to model):
  <tool_response>response:FUNC_NAME{key:value,key:<|"|>string_val<|"|>}<tool_response|>

This module handles:
1. Building tool declaration prompts in Gemma 4 format
2. Parsing <|tool_call> tags from model output
3. Building <tool_response> messages to feed results back
4. Fallback JSON parsing for non-native outputs
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict
    raw_text: str = ""


@dataclass
class ParseResult:
    """Result of parsing a model response for tool calls."""
    content: str
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    has_tools: bool = False
    # Diagnostics for round_end telemetry. `markers_seen` counts every
    # `<tool_call>`-shaped pattern the parser noticed BEFORE attempting
    # extraction; if it's > len(tool_calls) we had a malformed call the
    # model emitted but we silently dropped. `parse_errors` records the
    # specific failure reason for each drop (regex match without name,
    # JSON decode error, etc.) so the user-visible "model bailed mid-
    # turn" failure mode can be attributed to a parse drop instead of
    # being blamed on the model.
    markers_seen: int = 0
    parse_errors: list[str] = field(default_factory=list)

    def to_ollama_format(self) -> list[dict]:
        """Convert to Ollama-compatible tool_calls format."""
        return [
            {
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            }
            for tc in self.tool_calls
        ]


# ── Gemma 4 special token patterns ──────────────────────────────────────

# <|tool_call>call:FUNC_NAME{...}<tool_call|>
GEMMA4_TOOL_CALL_RE = re.compile(
    r'<\|tool_call\>call:(\w+)\{(.*?)\}<tool_call\|>',
    re.DOTALL,
)

# Also handle slight variations the model might produce
GEMMA4_TOOL_CALL_ALT_RE = re.compile(
    r'<\|tool_call\>\s*call\s*:\s*(\w+)\s*\{(.*?)\}\s*<\s*tool_call\s*\|>',
    re.DOTALL,
)

# Gemma 4 string delimiter: <|"|>value<|"|>
GEMMA4_STRING_RE = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)

# Fallback: standard JSON tool call tags
JSON_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*(.*?)\s*</tool_call>',
    re.DOTALL,
)

# Fallback: inline JSON
INLINE_TOOL_RE = re.compile(
    r'\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
)


def _parse_gemma4_args(args_str: str) -> dict:
    """Parse Gemma 4's custom argument format into a Python dict.

    Input format: param:<|"|>value<|"|>,param2:<|"|>value2<|"|>
    Also handles: param:123, param:true, param:[<|"|>a<|"|>,<|"|>b<|"|>]
    """
    result: dict = {}
    if not args_str.strip():
        return result

    # Replace Gemma 4 string delimiters with JSON-friendly quotes
    # <|"|>value<|"|> -> "value"
    jsonified = GEMMA4_STRING_RE.sub(r'"\1"', args_str)

    # Now parse key:value pairs
    # The format is like: key:value,key2:value2
    # where value can be "string", number, boolean, or [array]
    try:
        # Try wrapping in braces and parsing as JSON
        json_str = "{" + _kv_to_json(jsonified) + "}"
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Manual fallback parsing
    current_key = ""
    i = 0
    while i < len(jsonified):
        # Find key
        colon_pos = jsonified.find(":", i)
        if colon_pos == -1:
            break
        current_key = jsonified[i:colon_pos].strip().strip(",").strip()
        i = colon_pos + 1

        # Find value
        if i < len(jsonified) and jsonified[i] == '"':
            # Quoted string
            end = jsonified.find('"', i + 1)
            if end == -1:
                end = len(jsonified)
            result[current_key] = jsonified[i + 1:end]
            i = end + 1
        elif i < len(jsonified) and jsonified[i] == '[':
            # Array
            end = jsonified.find(']', i)
            if end == -1:
                end = len(jsonified)
            array_str = jsonified[i:end + 1]
            try:
                result[current_key] = json.loads(array_str)
            except json.JSONDecodeError:
                result[current_key] = array_str
            i = end + 1
        else:
            # Unquoted value (number, boolean)
            comma_pos = jsonified.find(",", i)
            if comma_pos == -1:
                val_str = jsonified[i:].strip()
            else:
                val_str = jsonified[i:comma_pos].strip()
                i = comma_pos + 1
            # Parse type
            if val_str.lower() == "true":
                result[current_key] = True
            elif val_str.lower() == "false":
                result[current_key] = False
            elif val_str.isdigit():
                result[current_key] = int(val_str)
            else:
                try:
                    result[current_key] = float(val_str)
                except ValueError:
                    result[current_key] = val_str
            if comma_pos != -1:
                continue
            break

    return result


def _kv_to_json(text: str) -> str:
    """Convert key:value,key2:value2 to "key":value,"key2":value2."""
    # Add quotes around keys that aren't already quoted
    result = re.sub(
        r'(?<!["\w])(\w+)\s*:',
        r'"\1":',
        text,
    )
    return result


# ── Main parser ──────────────────────────────────────────────────────────

def parse_tool_calls(text: str) -> ParseResult:
    """Parse tool calls from raw Gemma 4 model output.

    Tries formats in order:
    1. <|tool_call>call:NAME{args}<tool_call|> (Gemma 4 native)
    2. <tool_call>JSON</tool_call> (generic)
    3. Inline {"name": "...", "arguments": {...}}
    """
    tool_calls: list[ParsedToolCall] = []
    cleaned = text
    # Diagnostics — populated as the parser attempts each format so
    # downstream telemetry can distinguish "no tool call" from "tool
    # call we silently dropped." See ParseResult docstring.
    markers_seen = 0
    parse_errors: list[str] = []

    # NOTE on .strip() across this function: quantized models (Qwen 3.6
    # IQ2_M in particular) routinely emit tool names with leading or
    # trailing whitespace inside the JSON, like `"name": "list_files "`.
    # Without stripping at the parser, that trailing space propagates
    # all the way to dispatcher dict-lookups and produces opaque
    # KeyError('list_files ') errors. Three capture sites below all
    # need to .strip() the name as it's pulled out of the regex/JSON.

    # 1. Gemma 4 native format.
    #
    # Two regexes are tried, but we must NOT append matches from both:
    # the primary pattern and the whitespace-tolerant ALT pattern
    # overlap on the common case, which silently caused every tool
    # call to be dispatched twice (verified: parsing a single
    # `<|tool_call>call:write_file{...}<tool_call|>` produced two
    # identical ParsedToolCall entries → the agent ran write_file
    # twice per emitted call). Try the primary first; only fall through
    # to the ALT pattern on the same block of text if the primary
    # didn't match anything there.
    seen_spans: list[tuple[int, int]] = []
    for pattern in (GEMMA4_TOOL_CALL_RE, GEMMA4_TOOL_CALL_ALT_RE):
        for match in pattern.finditer(text):
            span = match.span()
            # Skip this match if it overlaps a span captured by a
            # previous (higher-priority) pattern.
            if any(not (span[1] <= s[0] or span[0] >= s[1]) for s in seen_spans):
                continue
            seen_spans.append(span)
            markers_seen += 1
            name = (match.group(1) or "").strip()
            args_raw = match.group(2)
            args = _parse_gemma4_args(args_raw)
            if not name:
                parse_errors.append("gemma4: empty tool name in match")
                continue
            tool_calls.append(ParsedToolCall(
                name=name,
                arguments=args,
                raw_text=match.group(0),
            ))
            cleaned = cleaned.replace(match.group(0), "")

    if tool_calls:
        return ParseResult(
            content=cleaned.strip(), tool_calls=tool_calls, has_tools=True,
            markers_seen=markers_seen, parse_errors=parse_errors,
        )

    # 2. Generic <tool_call>JSON</tool_call>
    for match in JSON_TOOL_CALL_RE.finditer(text):
        markers_seen += 1
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
            name = (data.get("name", "") or "").strip()
            args = data.get("arguments", data.get("parameters", {}))
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                tool_calls.append(ParsedToolCall(name=name, arguments=args, raw_text=raw))
                cleaned = cleaned.replace(match.group(0), "")
            else:
                parse_errors.append("json: tool_call JSON missing 'name' field")
        except json.JSONDecodeError as e:
            parse_errors.append(f"json: decode error — {str(e)[:80]}")
            continue

    if tool_calls:
        return ParseResult(
            content=cleaned.strip(), tool_calls=tool_calls, has_tools=True,
            markers_seen=markers_seen, parse_errors=parse_errors,
        )

    # 3. Inline JSON
    for match in INLINE_TOOL_RE.finditer(text):
        markers_seen += 1
        name = (match.group(1) or "").strip()
        try:
            args = json.loads(match.group(2))
            tool_calls.append(ParsedToolCall(name=name, arguments=args, raw_text=match.group(0)))
            cleaned = cleaned.replace(match.group(0), "")
        except json.JSONDecodeError as e:
            parse_errors.append(f"inline: decode error — {str(e)[:80]}")
            continue

    return ParseResult(
        content=cleaned.strip(), tool_calls=tool_calls, has_tools=bool(tool_calls),
        markers_seen=markers_seen, parse_errors=parse_errors,
    )


# ── Tool declaration builder (for MLX/HF prompts) ───────────────────────

def build_gemma4_tool_declaration(tool: dict) -> str:
    """Build a Gemma 4 format tool declaration from an Ollama-style tool schema.

    Input (Ollama format):
        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

    Output (Gemma 4 format):
        <|tool>declaration:NAME{description:<|"|>DESC<|"|>,parameters:{...}}<tool|>
    """
    func = tool.get("function", {})
    name = func.get("name", "")
    desc = func.get("description", "")
    params = func.get("parameters", {})

    # Build properties string
    properties = params.get("properties", {})
    required = params.get("required", [])

    prop_parts = []
    for pname, pinfo in properties.items():
        ptype = pinfo.get("type", "string")
        pdesc = pinfo.get("description", "")
        prop_parts.append(
            f'{pname}:{{description:<|"|>{pdesc}<|"|>,type:<|"|>{ptype}<|"|>}}'
        )

    props_str = ",".join(prop_parts)
    req_str = ",".join(f'<|"|>{r}<|"|>' for r in required)

    return (
        f'<|tool>declaration:{name}{{description:<|"|>{desc}<|"|>,'
        f'parameters:{{properties:{{{props_str}}},'
        f'required:[{req_str}],type:<|"|>object<|"|>}}}}<tool|>'
    )


def build_tool_response(name: str, result: str) -> str:
    """Build a Gemma 4 format tool response.

    Output: <tool_response>response:NAME{result:<|"|>RESULT<|"|>}<tool_response|>
    """
    # Escape the result for the special token format
    escaped = result.replace('<|"|>', '').replace('<tool_response|>', '')
    # Truncate very long results
    if len(escaped) > 4000:
        escaped = escaped[:4000] + "...[truncated]"
    return f'<tool_response>response:{name}{{result:<|"|>{escaped}<|"|>}}<tool_response|>'


def inject_tool_schemas_into_prompt(system_prompt: str, tools: list[dict]) -> str:
    """Inject tool schemas into the system prompt using Gemma 4's native format.

    For MLX/HF backends that can't receive tools via API, we inject the
    tool declarations directly into the system prompt using Gemma 4's
    special token format.
    """
    if not tools:
        return system_prompt

    declarations = [build_gemma4_tool_declaration(tool) for tool in tools]

    tool_block = (
        "\n\n" + "\n".join(declarations) + "\n\n"
        "When you need to use a tool, output a tool call in this exact format:\n"
        "<|tool_call>call:FUNCTION_NAME{param:<|\"|>value<|\"|>}<tool_call|>\n\n"
        "You can make multiple tool calls. After receiving tool results, "
        "use them to provide your final answer."
    )

    return system_prompt + tool_block
