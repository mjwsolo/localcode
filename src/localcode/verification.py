from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from .shell import run_shell


DEFAULT_VERIFY_COMMANDS = [
    "pytest -q",
    "python -m pytest -q",
    "npm test -- --runInBand",
    "pnpm test",
    "cargo test -q",
]


@dataclass
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
