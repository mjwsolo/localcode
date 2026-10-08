"""Typed decisions on the local model service, without a chat/tool loop."""
from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import sys
import urllib.error
import urllib.request
from pathlib import Path

MAX_BYTES = 8 * 1024 * 1024


class DecisionError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def supported(alias: str | None) -> bool:
    from .models_catalog import by_filename, group_for_filename
    name = (alias or "").removeprefix("localcode/")
    name = name if name.endswith(".gguf") else name + ".gguf"
    choice = by_filename(name)
    group = group_for_filename(name)
    return bool((choice and choice.supports_systemone) or (group and group.supports_systemone))


def build_request(question: str, kind: str, options: list[str], state="", images=None) -> dict:
    types = {"yes-no": "noul", "choice": "choice", "score": "score"}
    if kind not in types:
        raise DecisionError("Answer type must be yes-no, choice or score")
    item = {"type": types[kind], "instructions": question.strip()}
    labels = [s.strip() for s in options]
    if kind == "choice":
        if len(labels) != len(set(labels)):
            raise DecisionError("Each choice must have a different label")
        item["criteria"] = {s: None for s in labels}
    if kind == "score":
        item["criteria"] = labels
    if kind == "yes-no" and labels:
        raise DecisionError("Yes/no does not use choice labels")
    data = {"state": state, "questions": {"answer": item}}
    if images:
        data["images"] = images
    return validate_request(data)


def validate_request(data) -> dict:
    if not isinstance(data, dict) or set(data) - {"state", "questions", "images"}:
        raise DecisionError("Use state, questions and optional images")
    if not isinstance(data.get("state"), (str, dict, list)):
        raise DecisionError("Content must be text, a JSON object or a JSON array")
    questions = data.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 16:
        raise DecisionError("Supply between 1 and 16 questions")
    for key, item in questions.items():
        if not isinstance(key, str) or not key or len(key) > 128 or not isinstance(item, dict):
            raise DecisionError("Questions need non-empty IDs and question objects")
        if set(item) - {"type", "instructions", "criteria"}:
            raise DecisionError("Questions use type, instructions and criteria")
        instructions = item.get("instructions")
        if not isinstance(instructions, (str, dict, list)) or not instructions:
            raise DecisionError("A question is required")
        kind, criteria = item.get("type"), item.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 52:
                raise DecisionError("Choice needs 2 to 52 options")
            if any(not isinstance(label, str) or not label.strip() or len(label) > 512 or any(ord(c) < 32 for c in label) for label in criteria):
                raise DecisionError("Choice labels must be non-empty text without control characters")
            if any(value is not None and not isinstance(value, str) for value in criteria.values()):
                raise DecisionError("Choice descriptions must be text or null")
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10 or any(not isinstance(s, str) or not s.strip() for s in criteria):
                raise DecisionError("Score needs 2 to 10 levels, lowest to highest")
        elif kind == "noul":
            if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"} or any(not isinstance(v, str) for v in criteria.values())):
                raise DecisionError("Yes/no criteria use true and false descriptions")
        else:
            raise DecisionError("Unknown question type")
    images = data.get("images", [])
    if not isinstance(images, list) or len(images) > 1:
        raise DecisionError("OpenJev supports one image per decision request")
    for value in images:
        if not isinstance(value, str) or not value.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")):
            raise DecisionError("Images must be PNG, JPEG or WebP data URLs")
        try:
            base64.b64decode(value.split(",", 1)[1], validate=True)
        except ValueError as exc:
            raise DecisionError("Image data is not valid base64") from exc
    try:
        size = len(json.dumps(data, allow_nan=False).encode())
    except (TypeError, ValueError) as exc:
        raise DecisionError("Decision input must be valid JSON") from exc
    if size > MAX_BYTES:
        raise DecisionError("Decision input exceeds 8 MB", 413)
    return data


def validate_response(data, request) -> dict:
    def number(value, low=0, high=1):
        return type(value) in (float, int) and math.isfinite(value) and low <= value <= high

    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict) or set(answers) != set(request["questions"]):
        raise DecisionError("Decision server returned missing or unexpected answers", 502)
    for key, question in request["questions"].items():
        answer = answers[key]
        if not isinstance(answer, dict):
            raise DecisionError("Invalid decision answer", 502)
        kind = question["type"]
        if kind == "noul":
            if not number(answer.get("noul")):
                raise DecisionError("Invalid yes/no probability", 502)
            continue
        probs = answer.get("probabilities")
        labels = set(question["criteria"]) if kind == "choice" else {str(i) for i in range(len(question["criteria"]))}
        if not isinstance(probs, dict) or set(probs) != labels or not all(number(v) for v in probs.values()) or abs(sum(probs.values()) - 1) > 0.001:
            raise DecisionError("Invalid decision probabilities", 502)
        if kind == "choice" and answer.get("choice") not in labels:
            raise DecisionError("Decision answer is not one of the supplied choices", 502)
        if kind == "score" and not number(answer.get("score"), high=len(labels) - 1):
            raise DecisionError("Decision score is outside the supplied levels", 502)
    return data


def post(url: str, body: dict, headers: dict, timeout=60) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            message = json.load(exc).get("error", "Decision request failed")
            if isinstance(message, dict):
                message = message.get("message", "Decision request failed")
        except (ValueError, OSError):
            message = "Decision request failed"
        raise DecisionError(str(message), exc.code) from exc
    except (OSError, ValueError) as exc:
        raise DecisionError("The local decision service did not return a usable response", 502) from exc


def image_url(path: Path) -> str:
    kind = mimetypes.guess_type(path.name)[0]
    if kind not in {"image/png", "image/jpeg", "image/webp"}:
        raise DecisionError("Image must be PNG, JPEG or WebP")
    if path.stat().st_size > 5 * 1024 * 1024:
        raise DecisionError("Image exceeds 5 MB", 413)
    return "data:" + kind + ";base64," + base64.b64encode(path.read_bytes()).decode()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="localcode decide", description="Ask the loaded decision model a typed question; no coding tools are executed.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--type", choices=["yes-no", "choice", "score"], default="yes-no", dest="kind")
    parser.add_argument("--option", action="append", default=[], help="Choice label or ordered score level; repeat for each")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--state", default="", help="Content to evaluate (optional)")
    inputs.add_argument("--state-file", type=Path, help="Read content from a UTF-8 file")
    parser.add_argument("--image", type=Path, help="Optional image")
    args = parser.parse_args(argv)
    try:
        state = args.state_file.read_text() if args.state_file else args.state
        body = build_request(args.question, args.kind, args.option, state, [image_url(args.image)] if args.image else None)
        from .ui.auth import control_headers, read_auth_file
        from .ui import run_dir
        from .ui.ports import find_running
        running = find_running()
        credentials = read_auth_file(run_dir())
        if running is None or credentials is None:
            raise DecisionError("Start LocalCode and load OpenJev before using decide", 409)
        data = post(f"http://127.0.0.1:{running[0]}/decision", body, control_headers(credentials[0]))
        validate_response(data, body)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    except (DecisionError, OSError) as exc:
        print("localcode decide: " + str(exc), file=sys.stderr)
        return 1
