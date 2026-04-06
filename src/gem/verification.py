from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from .shell import run_shell


DEFAULT_VERIFY_COMMANDS = [
    "pytest -q",
    "python -m pytest -q",
    "npm test -- --runInBand",
    "pnpm test",
    "cargo test -q",
]


@dataclass(slots=True)
class VerificationStep:
    command: str
    label: str


def _has_any(repo_root: Path, *names: str) -> bool:
    return any((repo_root / name).exists() for name in names)


def guess_verify_command(repo_root: Path) -> str | None:
    if (repo_root / "pytest.ini").exists() or (repo_root / "pyproject.toml").exists():
        return "pytest -q"
    if (repo_root / "package.json").exists():
        return "npm test -- --runInBand"
    if (repo_root / "Cargo.toml").exists():
        return "cargo test -q"
    return None


def build_verification_plan(repo_root: Path, bias: str = "balanced") -> list[VerificationStep]:
    plan: list[VerificationStep] = []
    if _has_any(repo_root, "ruff.toml", ".ruff.toml"):
        plan.append(VerificationStep("ruff check .", "ruff"))
    if _has_any(repo_root, "pyproject.toml", "pytest.ini", "tox.ini"):
        if bias == "thorough":
            plan.append(VerificationStep("python -m compileall src", "compile"))
        plan.append(VerificationStep("pytest -q", "pytest"))
    if (repo_root / "package.json").exists():
        plan.append(VerificationStep("npm run lint -- --if-present", "lint"))
        if bias == "thorough":
            plan.append(VerificationStep("npm run build -- --if-present", "build"))
        plan.append(VerificationStep("npm test -- --runInBand", "test"))
    if (repo_root / "Cargo.toml").exists():
        plan.append(VerificationStep("cargo fmt --check", "fmt"))
        plan.append(VerificationStep("cargo test -q", "test"))
    if not plan:
        fallback = guess_verify_command(repo_root)
        if fallback:
            plan.append(VerificationStep(fallback, "fallback"))
    deduped: list[VerificationStep] = []
    seen: set[str] = set()
    for step in plan:
        if step.command in seen:
            continue
        deduped.append(step)
        seen.add(step.command)
    return deduped


def build_outcome_verification_plan(task: str, changed_files: list[str]) -> list[VerificationStep]:
    task_lower = task.lower()
    plan: list[VerificationStep] = []
    python_files = [f for f in changed_files if f.endswith(".py")]

    for file_name in python_files:
        plan.append(VerificationStep(f'python3 -m py_compile "{file_name}"', f"py_compile:{file_name}"))

    if any(token in task_lower for token in ("game", "app", "ui", "website", "dashboard")) and python_files:
        primary = python_files[0]
        plan.append(VerificationStep(
            f'python3 - <<\'PY\'\nimport ast, pathlib\npath = pathlib.Path("{primary}")\nast.parse(path.read_text())\nprint("ast ok:", path.name)\nPY',
            "ast-smoke",
        ))

    seen: set[str] = set()
    deduped: list[VerificationStep] = []
    for step in plan:
        if step.command in seen:
            continue
        deduped.append(step)
        seen.add(step.command)
    return deduped


def run_verification(repo_root: Path, command: str | None = None, bias: str = "balanced") -> tuple[str, int]:
    plan = [VerificationStep(command, "user")] if command else build_verification_plan(repo_root, bias=bias)
    if not plan:
        return "No verification command detected.", 1
    outputs: list[str] = []
    exit_code = 0
    for step in plan:
        chosen = step.command
        binary = chosen.split()[0]
        if shutil.which(binary) is None:
            outputs.append(f"[{step.label}] $ {chosen}\nSkipped: command not found")
            continue
        result = run_shell(chosen, str(repo_root))
        outputs.append(f"[{step.label}] $ {chosen}\n{result.output}")
        if result.returncode != 0:
            exit_code = result.returncode
            break
    return "\n\n".join(outputs), exit_code


def run_outcome_verification(repo_root: Path, task: str, changed_files: list[str]) -> tuple[str, int]:
    plan = build_outcome_verification_plan(task, changed_files)
    if not plan:
        return "No outcome verification steps detected.", 0
    outputs: list[str] = []
    exit_code = 0
    for step in plan:
        result = subprocess.run(
            step.command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=30,
        )
        outputs.append(f"[{step.label}] $ {step.command}\n{result.stdout or result.stderr}".rstrip())
        if result.returncode != 0:
            exit_code = result.returncode
            break
    return "\n\n".join(outputs), exit_code
