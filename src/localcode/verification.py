from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from ._subproc_env import clean_env
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


def classify_artifact(file_name: str, task: str = "") -> str:
    lower = file_name.lower()
    task_lower = task.lower()
    if lower.endswith(".py"):
        if any(token in task_lower for token in ("app", "game", "ui", "dashboard", "server", "cli")):
            return "python_app"
        return "python_module"
    if lower.endswith((".sh", ".bash")):
        return "shell_script"
    if lower.endswith((".html", ".css", ".js", ".ts", ".tsx", ".jsx")):
        return "web_asset"
    if lower.endswith((".json", ".toml", ".yaml", ".yml")):
        return "config"
    return "generic"


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
    for file_name in changed_files:
        artifact = classify_artifact(file_name, task)
        if artifact in {"python_app", "python_module"}:
            plan.append(VerificationStep(f'python3 -m py_compile "{file_name}"', f"py_compile:{file_name}"))
        if artifact == "python_app":
            plan.append(VerificationStep(
                f'python3 - <<\'PY\'\nimport ast, pathlib\npath = pathlib.Path("{file_name}")\nast.parse(path.read_text())\nprint("ast ok:", path.name)\nPY',
                f"ast-smoke:{file_name}",
            ))
            plan.append(VerificationStep(
                f'python3 - <<\'PY\'\nimport pathlib, runpy, os\nos.environ.setdefault("SDL_VIDEODRIVER", "dummy")\nos.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")\npath = pathlib.Path("{file_name}")\ncode = path.read_text()\nns = {{}}\nexec(compile(code, str(path), "exec"), ns, ns)\nprint("load ok:", path.name)\nPY',
                f"load-smoke:{file_name}",
            ))
        elif artifact == "shell_script":
            plan.append(VerificationStep(f'bash -n "{file_name}"', f"shellcheck-lite:{file_name}"))
        elif artifact == "web_asset":
            plan.append(VerificationStep(
                f'python3 - <<\'PY\'\nfrom pathlib import Path\npath = Path("{file_name}")\ntext = path.read_text(errors="replace")\nassert text.strip(), "empty file"\nprint("asset ok:", path.name)\nPY',
                f"asset-smoke:{file_name}",
            ))
        elif artifact == "config":
            plan.append(VerificationStep(
                f'python3 - <<\'PY\'\nfrom pathlib import Path\npath = Path("{file_name}")\ntext = path.read_text(errors="replace")\nassert text.strip(), "empty config"\nprint("config ok:", path.name)\nPY',
                f"config-smoke:{file_name}",
            ))
            if file_name.endswith("package.json"):
                package_dir = str(Path(file_name).parent)
                node_command = """cd "{dir}" && node - <<'NODE'
const fs = require('fs');
let browserslist;
try {{ browserslist = require('browserslist'); }} catch (err) {{
  console.log('package.json ok:', JSON.parse(fs.readFileSync('package.json', 'utf8')).name || 'unnamed');
  process.exit(0);
}}
const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
const config = browserslist.findConfig(process.cwd()) || pkg.browserslist;
if (config) {{
  const entries = Array.isArray(config) || typeof config === 'string'
    ? [config]
    : Object.values(config);
  for (const entry of entries) {{
    if (entry) browserslist(entry);
  }}
}}
console.log('package.json ok:', pkg.name || 'unnamed');
NODE""".format(dir=package_dir)
                plan.append(VerificationStep(node_command, f"package-json-smoke:{file_name}"))

    python_files = [f for f in changed_files if f.endswith(".py")]
    if any(token in task_lower for token in ("website", "dashboard", "frontend", "landing page")) and not python_files:
        for file_name in changed_files:
            if classify_artifact(file_name, task) == "web_asset":
                plan.append(VerificationStep(
                    f'python3 - <<\'PY\'\nfrom pathlib import Path\npath = Path("{file_name}")\nprint("preview bytes:", len(path.read_text(errors="replace")))\nPY',
                    f"web-preview:{file_name}",
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
            env=clean_env(),
        )
        outputs.append(f"[{step.label}] $ {step.command}\n{result.stdout or result.stderr}".rstrip())
        if result.returncode != 0:
            exit_code = result.returncode
            break
    return "\n\n".join(outputs), exit_code
