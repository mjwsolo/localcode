from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

from .app import GemApp
from .config import AppConfig
from .models import resolve_profile
from .performance import benchmark_report
from .runtime import GemRuntimeGateway
from .verification import run_outcome_verification, run_verification


@dataclass(slots=True)
class BenchmarkTask:
    name: str
    prompt: str
    verify_command: str | None = None
    expected_keywords: list[str] | None = None
    expected_files: list[str] | None = None
    min_total_lines: int = 0


def load_tasks(path: Path) -> list[BenchmarkTask]:
    data = json.loads(path.read_text())
    return [
        BenchmarkTask(
            name=item["name"],
            prompt=item["prompt"],
            verify_command=item.get("verify_command"),
            expected_keywords=item.get("expected_keywords"),
            expected_files=item.get("expected_files"),
            min_total_lines=int(item.get("min_total_lines", 0)),
        )
        for item in data
    ]


@contextmanager
def _isolated_gem_home(base_dir: Path):
    old = os.environ.get("GEM_HOME")
    home = base_dir / ".gem-home"
    os.environ["GEM_HOME"] = str(home)
    try:
        yield home
    finally:
        if old is None:
            os.environ.pop("GEM_HOME", None)
        else:
            os.environ["GEM_HOME"] = old


def _collect_workspace_files(root: Path) -> list[str]:
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        files.append(str(rel))
    return files


def _read_bundle(root: Path, files: list[str]) -> str:
    parts: list[str] = []
    for rel in files[:8]:
        full = root / rel
        try:
            text = full.read_text(errors="replace")
        except Exception:
            continue
        parts.append(f"## {rel}\n{text[:6000]}")
    return "\n\n".join(parts)


def _judge_quality(app: GemApp, task: BenchmarkTask, root: Path, files: list[str], verify_code: int) -> tuple[int, str]:
    if not files:
        return 0, "no files created"
    bundle = _read_bundle(root, files)
    if not bundle.strip():
        return 0, "unable to read outputs"
    try:
        response = app.engine.generate_once([
            {
                "role": "system",
                "content": (
                    "You are a strict local coding benchmark judge.\n"
                    "Return strict JSON only with keys: score, verdict, rationale.\n"
                    "A runnable but generic result should score modestly, not highly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{task.prompt}\n\n"
                    f"Verification exit code: {verify_code}\n\n"
                    f"Generated files:\n{bundle}\n\n"
                    "Score from 0-100 based on completeness, fidelity to the request, and practical usefulness."
                ),
            },
        ])
        match = re.search(r"\{.*\}", response, re.DOTALL)
        payload = json.loads(match.group(0) if match else response)
        return int(payload.get("score", 0) or 0), str(payload.get("rationale", "")).strip()[:240]
    except Exception as exc:
        return 0, f"judge failed: {exc}"


def _heuristic_score(task: BenchmarkTask, root: Path, files: list[str], verify_code: int) -> tuple[int, int, int]:
    bundle = _read_bundle(root, files).lower()
    keyword_hits = 0
    if task.expected_keywords:
        keyword_hits = sum(1 for keyword in task.expected_keywords if keyword.lower() in bundle)
    total_lines = 0
    for rel in files:
        try:
            total_lines += len((root / rel).read_text(errors="replace").splitlines())
        except Exception:
            pass
    score = 0
    if files:
        score += 35
    if verify_code == 0:
        score += 25
    if task.expected_keywords:
        score += min(20, keyword_hits * max(1, 20 // max(1, len(task.expected_keywords))))
    if task.min_total_lines:
        score += min(20, int(min(1.0, total_lines / max(1, task.min_total_lines)) * 20))
    return score, keyword_hits, total_lines


def _extract_first_token_s(memory: dict, runtime_mode: str) -> float | None:
    try:
        value = memory["runtime_telemetry"]["gemma4-26b-laptop"][runtime_mode]["ema_first_token_s"]
        return float(value)
    except Exception:
        return None


def _mute_app_output(app: GemApp) -> None:
    app.console.print = lambda *args, **kwargs: None

    app.out.set_event_callback(lambda *args, **kwargs: None)
    app.out.start_thinking = lambda *args, **kwargs: None
    app.out.start_streaming = lambda *args, **kwargs: None
    app.out.done = lambda *args, **kwargs: None
    app.out.set_error = lambda *args, **kwargs: None
    app.out.feed_thinking = lambda *args, **kwargs: None
    app.out.set_stage = lambda *args, **kwargs: None
    app.out.set_thinking_peek = lambda *args, **kwargs: None
    app.out.log_tool = lambda *args, **kwargs: 0
    app.out.tool_result = lambda *args, **kwargs: None
    app.out.stream = lambda *args, **kwargs: None
    app.out.print_info = lambda *args, **kwargs: None


def _run_single_task(
    config: AppConfig,
    workspace_root: Path,
    task: BenchmarkTask,
    profile_name: str | None,
    model_name: str | None,
) -> dict[str, object]:
    app = GemApp(config=config, cwd=workspace_root, profile_name=profile_name, model_name=model_name)
    try:
        _mute_app_output(app)
        before_memory = deepcopy(app._memory)
        started = time.perf_counter()
        answer = app.ask(task.prompt, stream=False)
        elapsed = time.perf_counter() - started
        files = task.expected_files or _collect_workspace_files(workspace_root)
        verify_output = ""
        verify_code = 0
        if task.verify_command:
            verify_output, verify_code = run_verification(workspace_root, task.verify_command, bias=app.profile.verification_bias)
        elif files:
            verify_output, verify_code = run_outcome_verification(workspace_root, task.prompt, files)
        heuristic_score, keyword_hits, total_lines = _heuristic_score(task, workspace_root, files, verify_code)
        quality_score, quality_rationale = _judge_quality(app, task, workspace_root, files, verify_code)
        runtime_mode = "fit" if config.runtime.provider == "llama_cpp" else "speed"
        memory_after = deepcopy(app._memory)
        first_token_s = _extract_first_token_s(memory_after, runtime_mode)
        prev_first = _extract_first_token_s(before_memory, runtime_mode)
        if first_token_s is not None and prev_first is not None and first_token_s == prev_first:
            first_token_s = None
        return {
            "name": task.name,
            "seconds": round(elapsed, 3),
            "first_token_s": round(first_token_s, 3) if first_token_s is not None else None,
            "chars": len(answer),
            "files": files,
            "verify_code": verify_code,
            "verify_output": verify_output[-1000:],
            "keyword_hits": keyword_hits,
            "total_lines": total_lines,
            "heuristic_score": heuristic_score,
            "quality_score": quality_score,
            "quality_rationale": quality_rationale,
        }
    finally:
        app.close()


def _apply_preset_in_memory(config: AppConfig, preset, model: str | None = None) -> AppConfig:
    config.runtime.provider = preset.runtime_provider
    config.runtime.profile = preset.profile
    if model:
        config.runtime.model = model
    config.runtime.max_context_chars = preset.max_context_chars
    config.runtime.mode = preset.mode
    config.runtime.planner_enabled = preset.planner_enabled
    config.runtime.adaptive_execution = preset.adaptive_execution
    config.runtime.planner_model = preset.planner_model
    config.runtime.draft_model = preset.draft_model
    config.runtime.quant_preset = preset.quant_preset
    config.runtime.cache_policy = preset.cache_policy
    config.runtime.rolling_window_messages = preset.rolling_window_messages
    config.runtime.llama_cpp_gpu_layers = preset.llama_cpp_gpu_layers
    config.runtime.llama_cpp_threads = preset.llama_cpp_threads
    config.runtime.llama_cpp_batch_size = preset.llama_cpp_batch_size
    config.runtime.llama_cpp_spec_type = preset.llama_cpp_spec_type
    config.runtime.llama_cpp_draft_max = preset.llama_cpp_draft_max
    config.runtime.llama_cpp_expert_offload = preset.llama_cpp_expert_offload
    config.runtime.llama_cpp_draft_model = preset.llama_cpp_draft_model
    config.runtime.llama_cpp_lookup_cache = preset.llama_cpp_lookup_cache
    config.runtime.kv_cache_type_k = preset.kv_cache_type_k
    config.runtime.kv_cache_type_v = preset.kv_cache_type_v
    config.runtime.low_overhead_mode = preset.low_overhead_mode
    config.runtime.laptop_26b_runtime_mode = preset.laptop_26b_runtime_mode
    config.runtime.execution_engine = "unified"
    return config


def _prepare_model_config(config: AppConfig, model_tag: str, mode: str, provider: str | None = None) -> AppConfig:
    run_config = deepcopy(config)
    if provider:
        run_config.runtime.provider = provider
    profile = resolve_profile(None, model_tag)
    run_config.runtime.profile = profile.key
    run_config.runtime.model = model_tag
    run_config.runtime.max_context_chars = min(
        max(8000, profile.recommended_context_chars),
        max(8000, run_config.runtime.max_context_chars),
    )
    run_config.runtime.low_overhead_mode = (profile.key == "gemma4-26b-laptop")
    run_config.runtime.planner_enabled = profile.key not in {"gemma4-e2b"}
    run_config.runtime.execution_engine = "unified"
    if profile.key == "gemma4-26b-laptop":
        _, preset = benchmark_report(run_config, mode)
        _apply_preset_in_memory(run_config, preset, model=model_tag)
    return run_config


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _llama_cpp_server(config: AppConfig, gguf_path: Path):
    run_config = deepcopy(config)
    run_config.runtime.provider = "llama_cpp"
    run_config.runtime.model = str(gguf_path)
    port = _free_port()
    run_config.runtime.base_url = f"http://127.0.0.1:{port}"
    gateway = GemRuntimeGateway(run_config.runtime)
    cmd = gateway.llama_server_command(str(gguf_path), port=port)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        text=True,
    )
    start = time.time()
    ready = False
    try:
        while time.time() - start < 60:
            ok, _ = GemRuntimeGateway(run_config.runtime).healthcheck()
            if ok:
                ready = True
                break
            time.sleep(1.0)
        if not ready:
            raise RuntimeError(f"llama.cpp server did not become ready for {gguf_path.name}")
        yield run_config
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except Exception:
            process.kill()


def run_task_benchmarks(
    config: AppConfig,
    repo_root: Path,
    task_file: Path,
    profile_name: str | None,
    model_name: str | None,
) -> list[dict[str, object]]:
    tasks = load_tasks(task_file)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="gem-bench-") as temp_dir:
        temp_root = Path(temp_dir)
        with _isolated_gem_home(temp_root):
            for task in tasks:
                task_root = temp_root / task.name.replace(" ", "_")
                task_root.mkdir(parents=True, exist_ok=True)
                task_config = deepcopy(config)
                rows.append(_run_single_task(task_config, task_root, task, profile_name, model_name))
    return rows


def compare_laptop_runtime_modes(
    config: AppConfig,
    repo_root: Path,
    task_file: Path,
    mode: str = "balanced",
    runtime_modes: list[str] | None = None,
    repeats: int = 1,
) -> list[dict[str, object]]:
    tasks = load_tasks(task_file)
    runtime_modes = runtime_modes or ["speed", "fit"]
    rows: list[dict[str, object]] = []

    for runtime_mode in runtime_modes:
        for repeat in range(repeats):
            run_config = deepcopy(config)
            run_config.runtime.laptop_26b_runtime_mode = runtime_mode
            _, preset = benchmark_report(run_config, mode)
            _apply_preset_in_memory(run_config, preset, model=run_config.runtime.model)

            with tempfile.TemporaryDirectory(prefix=f"gem-bench-{runtime_mode}-") as temp_dir:
                temp_root = Path(temp_dir)
                with _isolated_gem_home(temp_root):
                    for task in tasks:
                        task_root = temp_root / f"{task.name.replace(' ', '_')}_{repeat + 1}"
                        task_root.mkdir(parents=True, exist_ok=True)
                        result = _run_single_task(run_config, task_root, task, run_config.runtime.profile, run_config.runtime.model)
                        result.update(
                            {
                                "runtime_mode": runtime_mode,
                                "provider": run_config.runtime.provider,
                                "profile": run_config.runtime.profile,
                                "repeat": repeat + 1,
                            }
                        )
                        rows.append(result)
    return rows


def compare_explicit_models(
    config: AppConfig,
    repo_root: Path,
    task_file: Path,
    model_tags: list[str],
    mode: str = "balanced",
    provider: str | None = None,
    repeats: int = 1,
) -> list[dict[str, object]]:
    tasks = load_tasks(task_file)
    rows: list[dict[str, object]] = []

    for model_tag in model_tags:
        for repeat in range(repeats):
            run_config = _prepare_model_config(config, model_tag, mode=mode, provider=provider)
            with tempfile.TemporaryDirectory(prefix="gem-model-bench-") as temp_dir:
                temp_root = Path(temp_dir)
                with _isolated_gem_home(temp_root):
                    for task in tasks:
                        task_root = temp_root / f"{task.name.replace(' ', '_')}_{repeat + 1}"
                        task_root.mkdir(parents=True, exist_ok=True)
                        result = _run_single_task(
                            run_config,
                            task_root,
                            task,
                            run_config.runtime.profile,
                            model_tag,
                        )
                        result.update(
                            {
                                "model": model_tag,
                                "provider": run_config.runtime.provider,
                                "profile": run_config.runtime.profile,
                                "repeat": repeat + 1,
                            }
                        )
                        rows.append(result)
    return rows


def compare_gguf_models(
    config: AppConfig,
    repo_root: Path,
    task_file: Path,
    gguf_paths: list[Path],
    mode: str = "balanced",
    repeats: int = 1,
) -> list[dict[str, object]]:
    tasks = load_tasks(task_file)
    rows: list[dict[str, object]] = []

    for gguf_path in gguf_paths:
        for repeat in range(repeats):
            with _llama_cpp_server(config, gguf_path) as run_config:
                run_config.runtime.mode = mode
                profile = resolve_profile("gemma4-26b-laptop", str(gguf_path))
                run_config.runtime.profile = profile.key
                with tempfile.TemporaryDirectory(prefix="gem-gguf-bench-") as temp_dir:
                    temp_root = Path(temp_dir)
                    with _isolated_gem_home(temp_root):
                        for task in tasks:
                            task_root = temp_root / f"{task.name.replace(' ', '_')}_{repeat + 1}"
                            task_root.mkdir(parents=True, exist_ok=True)
                            result = _run_single_task(
                                run_config,
                                task_root,
                                task,
                                run_config.runtime.profile,
                                str(gguf_path),
                            )
                            result.update(
                                {
                                    "model": gguf_path.name,
                                    "provider": "llama_cpp",
                                    "profile": run_config.runtime.profile,
                                    "repeat": repeat + 1,
                                }
                            )
                            rows.append(result)
    return rows
