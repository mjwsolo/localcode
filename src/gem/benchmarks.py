from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

from .app import GemApp
from .config import AppConfig
from .verification import run_verification


@dataclass(slots=True)
class BenchmarkTask:
    name: str
    prompt: str
    verify_command: str | None = None
    expected_keywords: list[str] | None = None


def load_tasks(path: Path) -> list[BenchmarkTask]:
    data = json.loads(path.read_text())
    return [
        BenchmarkTask(
            name=item["name"],
            prompt=item["prompt"],
            verify_command=item.get("verify_command"),
            expected_keywords=item.get("expected_keywords"),
        )
        for item in data
    ]


def run_task_benchmarks(config: AppConfig, repo_root: Path, task_file: Path, profile_name: str | None, model_name: str | None) -> list[dict[str, object]]:
    tasks = load_tasks(task_file)
    app = GemApp(config=config, cwd=repo_root, profile_name=profile_name, model_name=model_name)
    rows: list[dict[str, object]] = []
    try:
        for task in tasks:
            started = time.perf_counter()
            answer = app.ask(task.prompt, stream=False)
            elapsed = time.perf_counter() - started
            keyword_hits = 0
            if task.expected_keywords:
                lowered = answer.lower()
                keyword_hits = sum(1 for keyword in task.expected_keywords if keyword.lower() in lowered)
            verify_output = ""
            verify_code = 0
            if task.verify_command:
                verify_output, verify_code = run_verification(repo_root, task.verify_command, bias=app.profile.verification_bias)
            rows.append(
                {
                    "name": task.name,
                    "seconds": round(elapsed, 3),
                    "chars": len(answer),
                    "keyword_hits": keyword_hits,
                    "verify_code": verify_code,
                    "verify_output": verify_output[-1000:],
                }
            )
    finally:
        app.close()
    return rows
