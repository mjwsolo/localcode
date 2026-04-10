from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable, TypeVar

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .browser import browser_status, ensure_browser_mcp
from .config import AppConfig, get_config_path, init_config_file, save_config
from .models import ALIASES, GEMMA_PROFILES, MLX_MODEL_IDS, get_runtime_model, resolve_profile
from .performance import MachineProfile, PerformancePreset, apply_preset, benchmark_report, should_promote_legacy_default_to_laptop_26b
from .runtime import GemRuntimeGateway


def provider_readiness(runtime_config) -> tuple[bool, list[str]]:
    """Stub: assume provider is ready after provider_checks module removal."""
    return True, []


def browser_voice_readiness(config) -> tuple[bool, list[str]]:
    """Stub: assume browser/voice ready after provider_checks module removal."""
    return True, []


def recommend_for_model_tag(model_tag: str):
    """Stub for removed model_recommend module."""
    from dataclasses import dataclass

    @dataclass
    class _Rec:
        model_tag: str
        backend: str
        quant_preset: str
        model_id_field: str
        note: str

    return _Rec(
        model_tag=model_tag,
        backend="ollama",
        quant_preset="balanced",
        model_id_field="model",
        note="model_recommend module removed",
    )


def runtime_command(runtime_config) -> str | None:
    """Stub: no auto-generated runtime command after runtime_launch removal."""
    return None

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class InstallPlan:
    label: str
    command: list[str]


@dataclass(frozen=True, slots=True)
class SetupStep:
    key: str
    phase: str
    detail: str


PROFILE_HINTS = {
    "gemma4-e2b": "faster for small machines",
    "gemma4-e4b": "balanced default for most laptops",
    "gemma4-26b-laptop": "best local quality/speed tradeoff on Apple Silicon laptops",
    "gemma4-26b-moe": "stronger coding on bigger local rigs",
    "gemma4-31b": "best quality for large local workstations",
}


def normalize_voice_stack(config: AppConfig, console: Console) -> None:
    if sys.version_info >= (3, 13) and config.voice.tts_provider == "kokoro":
        config.voice.tts_provider = "piper"
        console.print("Switched voice TTS from kokoro to piper because Python 3.13 is a bad fit for the current kokoro install stack.")
        save_config(config)


def _reason(text: str) -> str:
    parts = [chunk.strip() for chunk in text.split(".") if chunk.strip()]
    return "\n".join(f"- {part}" for part in parts[:2]) or f"- {text.strip()}"


def _progress_lines(steps: list[SetupStep], current_key: str) -> str:
    current_index = next((idx for idx, step in enumerate(steps) if step.key == current_key), 0)
    rows: list[str] = []
    for idx, step in enumerate(steps, start=1):
        if idx - 1 < current_index:
            marker = "[green]done [/green]"
        elif idx - 1 == current_index:
            marker = "[bold bright_green]now  [/bold bright_green]"
        else:
            marker = "[dim]next [/dim]"
        rows.append(f"{marker} {idx}/{len(steps)} {step.phase}")
    return "\n".join(rows)


MINER_FRAMES = [
    "  [green]/[/]\n [green]/[/]\n[green]/[/]\n",
    " [green]/[/]\n[green]/[/]\n[green]*[/]\n",
    "[green]/[/]\n[green]*[/]\n \n",
    " [green]/[/]\n[green]/[/]\n[green]*[/]\n",
]


def _runner_panel(label: str, detail: str, tick: int, steps: list[SetupStep], current_key: str) -> Panel:
    miner = MINER_FRAMES[tick % len(MINER_FRAMES)]
    current_index = next((idx for idx, step in enumerate(steps) if step.key == current_key), 0)
    brand = None  # no branding in bootstrap — app banner handles it
    status = Panel.fit(
        f"{miner}\n"
        f"[bold]phase[/bold]: {label}\n"
        f"[bold]step[/bold]:  {current_index + 1}/{len(steps)}\n"
        f"[bold]now[/bold]:   {detail}\n\n"
        f"{_progress_lines(steps, current_key)}\n\n"
        f"[dim]LocalCode is active. Press Ctrl-C to cancel.[/dim]",
        title="[green]Status[/green]",
        border_style="bright_green",
    )
    return Panel.fit(status, border_style="green")


_progress_status: dict[str, str] = {}


def run_with_runner(console: Console, step: SetupStep, steps: list[SetupStep], fn: Callable[[], T]) -> T:
    result: dict[str, T] = {}
    error: dict[str, BaseException] = {}
    _progress_status["line"] = ""

    def worker() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # pragma: no cover
            error["value"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    tick = 0
    with Live(_runner_panel(step.phase, step.detail, tick, steps, step.key), console=console, refresh_per_second=6) as live:
        while thread.is_alive():
            tick += 1
            detail = step.detail
            progress_line = _progress_status.get("line", "")
            if progress_line:
                detail = f"{step.detail}\n[dim]{progress_line}[/dim]"
            live.update(_runner_panel(step.phase, detail, tick, steps, step.key))
            time.sleep(0.16)
    _progress_status["line"] = ""
    if "value" in error:
        raise error["value"]
    return result["value"]


def _set_progress(line: str) -> None:
    """Callback for streaming progress into the runner panel."""
    _progress_status["line"] = line.strip()[-80:]  # keep last 80 chars


def detect_install_plan() -> InstallPlan | None:
    system = platform.system().lower()
    if system == "darwin" and shutil.which("brew"):
        return InstallPlan("Homebrew", ["brew", "install", "--cask", "ollama"])
    if system == "linux":
        if shutil.which("apt-get"):
            return InstallPlan("apt", ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
        if shutil.which("dnf"):
            return InstallPlan("dnf", ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
    return None


def detect_llama_cpp_install_plan() -> InstallPlan | None:
    system = platform.system().lower()
    if system == "darwin" and shutil.which("brew"):
        return InstallPlan("Homebrew", ["brew", "install", "llama.cpp"])
    if system == "linux" and shutil.which("pipx"):
        return InstallPlan("pipx", ["pipx", "install", "llama-cpp-python"])
    return None


def _find_turboquant_source() -> Path | None:
    """Locate the TurboQuant llama.cpp fork source directory."""
    # Check relative to this package (repo checkout)
    pkg_dir = Path(__file__).resolve().parent.parent.parent  # src/gem -> src -> repo root
    candidate = pkg_dir / "llama-cpp-turboquant"
    if (candidate / "CMakeLists.txt").exists():
        return candidate
    # Check home directory
    home_candidate = Path.home() / "llama-cpp-turboquant"
    if (home_candidate / "CMakeLists.txt").exists():
        return home_candidate
    return None


def _turboquant_binary_path() -> Path | None:
    """Return path to the built TurboQuant llama-server binary, or None if not built."""
    # Check data dir first (pre-built binary downloaded by bootstrap)
    data_dir = Path.home() / ".local" / "share" / "localcode"
    data_binary = data_dir / "llama-server"
    if data_binary.exists():
        return data_binary
    # Check source build
    source = _find_turboquant_source()
    if source is None:
        return None
    binary = source / "build" / "bin" / "llama-server"
    if binary.exists():
        return binary
    return None


_BINARY_RELEASE_URL = "https://github.com/mjwsolo/localcode/releases/download/v{version}/llama-server-{platform}"

def download_turboquant_binary(on_progress: Callable[[str], None] | None = None) -> tuple[bool, str]:
    """Download pre-built llama-server binary from GitHub Releases."""
    import urllib.request
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine == "arm64":
        plat = "macos-arm64"
    elif system == "linux" and machine in ("x86_64", "amd64"):
        plat = "linux-x86_64"
    else:
        return False, f"No pre-built binary for {system}-{machine}. Clone the repo and build from source."

    version = "0.1.0"
    url = _BINARY_RELEASE_URL.format(version=version, platform=plat)
    data_dir = Path.home() / ".local" / "share" / "localcode"
    data_dir.mkdir(parents=True, exist_ok=True)
    binary = data_dir / "llama-server"

    if on_progress:
        on_progress(f"downloading llama-server for {plat}...")
    try:
        urllib.request.urlretrieve(url, str(binary))
        binary.chmod(0o755)
        return True, str(binary)
    except Exception as e:
        return False, f"Download failed: {e}\nBuild from source instead: clone llama-cpp-turboquant and run cmake."


def _ensure_cmake() -> bool:
    """Install cmake if not present. Returns True if cmake is available."""
    if shutil.which("cmake"):
        return True
    system = platform.system().lower()
    if system == "darwin" and shutil.which("brew"):
        result = subprocess.run(["brew", "install", "cmake"], capture_output=True, text=True, check=False)
        return result.returncode == 0
    if system == "linux":
        for mgr, cmd in [("apt-get", ["sudo", "apt-get", "install", "-y", "cmake"]),
                         ("dnf", ["sudo", "dnf", "install", "-y", "cmake"])]:
            if shutil.which(mgr):
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                return result.returncode == 0
    return False


def _ensure_ollama() -> bool:
    """Install Ollama if not present. Returns True if Ollama is available."""
    if is_ollama_installed():
        return True
    plan = detect_install_plan()
    if plan is None:
        return False
    result = subprocess.run(plan.command, capture_output=True, text=True, check=False)
    return result.returncode == 0


def ensure_model_downloaded(model_tag: str, on_progress: Callable[[str], None] | None = None) -> tuple[bool, str]:
    """Ensure the model is downloaded via Ollama. Installs Ollama if needed."""
    if not _ensure_ollama():
        return False, "Could not install Ollama. Install manually: https://ollama.com/download"
    # Start Ollama service if not running
    if platform.system().lower() == "darwin":
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        start_new_session=True)
        import time; time.sleep(2)
    return pull_model(model_tag, on_progress=on_progress)


def build_turboquant(on_progress: Callable[[str], None] | None = None) -> tuple[bool, str]:
    """Build the TurboQuant llama.cpp fork from source with Metal support."""
    source = _find_turboquant_source()
    if source is None:
        return False, "TurboQuant source not found. Expected at llama-cpp-turboquant/ in repo root."
    if on_progress:
        on_progress("checking cmake...")
    if not _ensure_cmake():
        return False, "cmake is required and could not be auto-installed. Install with: brew install cmake"
    build_dir = source / "build"
    build_dir.mkdir(exist_ok=True)
    # Configure
    if on_progress:
        on_progress("configuring cmake...")
    result = subprocess.run(
        ["cmake", "..", "-DGGML_METAL=ON", "-DCMAKE_BUILD_TYPE=Release",
         "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"],
        cwd=str(build_dir), capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return False, f"cmake configure failed:\n{result.stderr[-500:]}"
    # Build
    import os
    jobs = str(min(os.cpu_count() or 4, 10))
    if on_progress:
        on_progress(f"compiling with {jobs} threads...")
    result = subprocess.run(
        ["cmake", "--build", ".", "--config", "Release", "-j", jobs],
        cwd=str(build_dir), capture_output=True, text=True, check=False,
        timeout=600,  # 10 min max
    )
    if result.returncode != 0:
        return False, f"build failed:\n{result.stderr[-500:]}"
    binary = build_dir / "bin" / "llama-server"
    if not binary.exists():
        return False, f"Build completed but llama-server binary not found at {binary}"
    return True, str(binary)


def is_ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def install_ollama(console: Console) -> tuple[bool, str]:
    plan = detect_install_plan()
    if plan is None:
        return False, "Automatic Ollama install is not available on this system."
    result = subprocess.run(plan.command, capture_output=True, text=True, check=False)
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0:
        return False, output or "Install command failed."
    return True, output or f"Installed via {plan.label}."


def install_llama_cpp(console: Console) -> tuple[bool, str]:
    plan = detect_llama_cpp_install_plan()
    if plan is None:
        return False, "Automatic llama.cpp install is not available on this system."
    result = subprocess.run(plan.command, capture_output=True, text=True, check=False)
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0:
        return False, output or "llama.cpp install command failed."
    return True, output or f"Installed via {plan.label}."


def detect_llama_cpp_server_command() -> str | None:
    for candidate in ("llama-server", "llama_cpp.server"):
        if shutil.which(candidate):
            return candidate
    return None


def pull_model(model_name: str, on_progress: Callable[[str], None] | None = None) -> tuple[bool, str]:
    """Pull an Ollama model with streaming progress output."""
    if not is_ollama_installed():
        return False, "Ollama is not installed."
    try:
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                if on_progress:
                    on_progress(line)
        returncode = process.wait(timeout=600)  # 10 min max
        output = "\n".join(lines[-5:])  # keep last 5 lines
        return returncode == 0, output or f"Pulled {model_name}."
    except subprocess.TimeoutExpired:
        process.kill()
        return False, f"Model pull timed out after 10 minutes. Try: ollama pull {model_name}"
    except Exception as exc:
        return False, f"Pull failed: {exc}"


def install_python_packages(packages: list[str]) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *packages],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, output or f"Installed {' '.join(packages)}."


def install_runtime_dependencies(provider: str) -> tuple[bool, str]:
    if provider == "mlx-local":
        return install_python_packages(["-U", "mlx-lm"])
    if provider == "huggingface-local":
        return install_python_packages(["transformers", "torch", "accelerate"])
    return False, "No provider-specific Python dependencies required."


def fetch_provider_model(provider: str, config: AppConfig, resolved_model: str) -> tuple[bool, str]:
    if provider == "mlx-local":
        model_id = config.runtime.mlx_model_id or resolved_model
        if not model_id:
            return False, "No MLX model id configured."
        script = (
            "from mlx_lm import load; import sys; "
            "load(sys.argv[1]); "
            "print(f'MLX model ready: {sys.argv[1]}')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, model_id],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        return result.returncode == 0, output or f"MLX model ready: {model_id}"
    if provider == "huggingface-local":
        model_id = config.runtime.huggingface_model_id or resolved_model
        if not model_id:
            return False, "No Hugging Face model id configured."
        script = (
            "from transformers import AutoTokenizer, AutoModelForCausalLM; import sys; "
            "mid=sys.argv[1]; "
            "AutoTokenizer.from_pretrained(mid); "
            "AutoModelForCausalLM.from_pretrained(mid); "
            "print(f'HF model ready: {mid}')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, model_id],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        return result.returncode == 0, output or f"HF model ready: {model_id}"
    return False, "No provider-specific model fetch action available."


def validate_provider(console: Console, config: AppConfig) -> bool:
    ok, messages = provider_readiness(config.runtime)
    if messages:
        for message in messages:
            console.print(message)
    return ok


def choose_profile(console: Console, current_profile: str) -> str:
    console.print("Choose a Gemma profile:")
    ordered_keys = ["gemma4-e2b", "gemma4-e4b", "gemma4-26b-laptop", "gemma4-26b-moe", "gemma4-31b"]
    for index, key in enumerate(ordered_keys, start=1):
        profile = GEMMA_PROFILES[key]
        hint = PROFILE_HINTS[key]
        marker = " default" if key == current_profile else ""
        short = profile.key.replace("gemma4-", "")
        console.print(f"  {index}. {short} [{hint}]{marker}")
    raw = input(f"Select profile [1-{len(ordered_keys)}] or press Enter for {current_profile}: ").strip()
    if not raw:
        return current_profile
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(ordered_keys):
            return ordered_keys[idx - 1]
    lowered = raw.lower()
    if lowered in ALIASES:
        return ALIASES[lowered]
    if lowered in GEMMA_PROFILES:
        return lowered
    console.print(f"Unknown choice '{raw}', using {current_profile}.")
    return current_profile


def choose_runtime_provider(console: Console, machine: MachineProfile, preset: PerformancePreset, current_provider: str) -> str:
    console.print("Choose a local runtime:")
    choices = [
        ("ollama", "easiest setup"),
        ("mlx-local", "best Apple Silicon path for MLX quantized Gemma"),
        ("llama_cpp", "best GGUF / tuned performance path"),
        ("huggingface-local", "advanced custom local checkpoints"),
    ]
    if machine.system != "darwin":
        choices = [item for item in choices if item[0] != "mlx-local"]
    default = preset.runtime_provider or current_provider
    for index, (key, label) in enumerate(choices, start=1):
        marker = " default" if key == default else ""
        console.print(f"  {index}. {key} [{label}]{marker}")
    raw = input(f"Select runtime [1-{len(choices)}] or press Enter for {default}: ").strip()
    if not raw:
        return default
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(choices):
            return choices[idx - 1][0]
    lowered = raw.lower()
    for key, _label in choices:
        if lowered == key:
            return key
    console.print(f"Unknown choice '{raw}', using {default}.")
    return default


def prompt_model_tag(console: Console, provider: str, current_model: str) -> str:
    hints = {
        "ollama": "example: gemma4:e4b",
        "mlx-local": "example: mlx-community/gemma-4-31b-it-4bit",
        "llama_cpp": "example: gemma4-27b-it-Q4_K_M.gguf",
        "huggingface-local": "example: google/gemma-4-27b-it-awq",
    }
    prompt = f"Optional model tag for {provider} [{hints.get(provider, 'leave empty for default')}]"
    raw = input(f"{prompt} or press Enter to keep {current_model or 'the profile default'}: ").strip()
    return raw or current_model


def _apply_model_routing(config: AppConfig, model_name: str) -> None:
    recommendation = recommend_for_model_tag(model_name)
    config.runtime.provider = recommendation.backend
    config.runtime.quant_preset = recommendation.quant_preset
    if recommendation.model_id_field == "mlx_model_id":
        config.runtime.mlx_model_id = model_name
        config.runtime.huggingface_model_id = ""
    elif recommendation.model_id_field == "huggingface_model_id":
        config.runtime.huggingface_model_id = model_name
        config.runtime.mlx_model_id = ""
    else:
        config.runtime.mlx_model_id = ""
        config.runtime.huggingface_model_id = ""


def run_runtime_wizard(
    console: Console,
    config: AppConfig,
    profile_name: str | None,
    model_name: str | None,
    benchmark: bool,
    assume_defaults: bool = False,
) -> tuple[str, str | None]:
    machine, preset = benchmark_report(config)
    if should_promote_legacy_default_to_laptop_26b(config, machine):
        apply_preset(config, preset, model=model_name or config.runtime.model)
    if benchmark:
        apply_preset(config, preset, model=model_name or config.runtime.model)
        console.print(f"Detected machine tier: {machine.tier}")
        console.print(f"Recommended mode: {preset.mode}")
        if preset.profile == "gemma4-26b-laptop":
            console.print("26B laptop mode: automatic Apple Silicon runtime selection enabled.")
        for note in preset.notes:
            console.print(f"- {note}")
    if assume_defaults:
        provider = preset.runtime_provider or config.runtime.provider
        config.runtime.provider = provider
        selected_profile = profile_name or config.runtime.profile or preset.profile
        chosen_model = model_name or config.runtime.model
        if chosen_model:
            _apply_model_routing(config, chosen_model)
        else:
            config.runtime.provider = provider
        return selected_profile, chosen_model or None
    provider = choose_runtime_provider(console, machine, preset, config.runtime.provider)
    config.runtime.provider = provider
    selected_profile = profile_name or choose_profile(console, config.runtime.profile)
    chosen_model = model_name or prompt_model_tag(console, provider, config.runtime.model)
    if chosen_model:
        _apply_model_routing(config, chosen_model)
    else:
        config.runtime.provider = provider
    return selected_profile, chosen_model or None


def run_setup(
    config: AppConfig,
    profile_name: str | None,
    model_name: str | None,
    auto_install: bool,
    benchmark: bool = False,
    assume_defaults: bool = False,
) -> int:
    console = Console()
    init_config_file()
    setup_steps = [
        SetupStep("runtime-install", "installing runtime", "Installing or preparing the selected local runtime."),
        SetupStep("runtime-check", "checking runtime", f"Checking {config.runtime.provider} readiness."),
        SetupStep("model-prepare", "preparing model", "Making sure the selected local model is ready."),
    ]
    selected_profile_name, chosen_model_name = run_runtime_wizard(
        console,
        config,
        profile_name,
        model_name,
        benchmark,
        assume_defaults=assume_defaults,
    )
    profile = resolve_profile(selected_profile_name, chosen_model_name or config.runtime.model)
    resolved_model = get_runtime_model(profile, chosen_model_name or config.runtime.model)
    config.runtime.profile = profile.key
    config.runtime.model = resolved_model
    normalize_voice_stack(config, console)
    if config.runtime.provider == "mlx-local" and not config.runtime.mlx_model_id:
        # Auto-pick the best MLX model for this profile
        config.runtime.mlx_model_id = MLX_MODEL_IDS.get(profile.key, resolved_model)
    if config.runtime.provider == "huggingface-local" and not config.runtime.huggingface_model_id:
        config.runtime.huggingface_model_id = resolved_model
    save_config(config)

    if auto_install and config.runtime.provider == "ollama" and not is_ollama_installed():
        install_step = SetupStep("runtime-install", "installing runtime", "Installing Ollama for local Gemma use.")
        ok, details = run_with_runner(console, install_step, setup_steps, lambda: install_ollama(console))
        if not ok:
            console.print(details)
            return 1

    if auto_install and config.runtime.provider in {"mlx-local", "huggingface-local"}:
        install_step = SetupStep("runtime-install", "installing runtime", f"Installing Python dependencies for {config.runtime.provider}.")
        ok, details = run_with_runner(console, install_step, setup_steps, lambda: install_runtime_dependencies(config.runtime.provider))
        console.print(details)
        if not ok:
            return 1
        validate_provider(console, config)

    if auto_install and config.runtime.provider == "llama_cpp":
        # Check if TurboQuant binary already exists (built or downloaded)
        if _turboquant_binary_path():
            binary_path = str(_turboquant_binary_path())
            if config.runtime.llama_cpp_binary != binary_path:
                config.runtime.llama_cpp_binary = binary_path
                save_config(config)
        # Try building from source first (repo checkout), then download pre-built binary
        elif _find_turboquant_source():
            install_step = SetupStep("runtime-install", "building TurboQuant", "Building TurboQuant llama.cpp fork with Metal support.")
            ok, binary_path = run_with_runner(console, install_step, setup_steps, lambda: build_turboquant(on_progress=_set_progress))
            if ok:
                config.runtime.llama_cpp_binary = binary_path
                save_config(config)
                console.print(f"TurboQuant built: {binary_path}")
            else:
                console.print(f"TurboQuant build failed: {binary_path}")
                console.print("Trying pre-built binary download...")
                install_step = SetupStep("runtime-install", "downloading server", "Downloading pre-built llama-server binary.")
                ok, binary_path = run_with_runner(console, install_step, setup_steps, lambda: download_turboquant_binary(on_progress=_set_progress))
                if ok:
                    config.runtime.llama_cpp_binary = binary_path
                    save_config(config)
                else:
                    console.print(f"Download failed: {binary_path}")
        else:
            # No source available (pip install) — download pre-built binary
            install_step = SetupStep("runtime-install", "downloading server", "Downloading pre-built llama-server binary.")
            ok, binary_path = run_with_runner(console, install_step, setup_steps, lambda: download_turboquant_binary(on_progress=_set_progress))
            if ok:
                config.runtime.llama_cpp_binary = binary_path
                save_config(config)
                console.print(f"Server downloaded: {binary_path}")
            else:
                console.print(f"Download failed: {binary_path}")
                console.print("Falling back to stock llama.cpp...")
                install_step = SetupStep("runtime-install", "installing runtime", "Installing llama.cpp where supported.")
                ok, details = run_with_runner(console, install_step, setup_steps, lambda: install_llama_cpp(console))
                console.print(details)
            console.print(details)

    # For llama_cpp provider, we still need Ollama to download the model GGUF
    if auto_install and config.runtime.provider == "llama_cpp":
        if not is_ollama_installed():
            install_step = SetupStep("runtime-install", "installing Ollama", "Installing Ollama for model download.")
            ok, details = run_with_runner(console, install_step, setup_steps, lambda: install_ollama(console))
            if not ok:
                console.print(details)
                console.print("Install Ollama manually: https://ollama.com/download")
        if is_ollama_installed():
            # Start Ollama service if not running
            import time as _time
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            _time.sleep(2)
            model_step = SetupStep("model-prepare", "downloading model", f"Pulling {resolved_model} via Ollama.")
            pulled, pull_details = run_with_runner(console, model_step, setup_steps, lambda: pull_model(resolved_model, on_progress=_set_progress))
            console.print(pull_details)

    engine = GemRuntimeGateway(config.runtime)
    runtime_step = SetupStep("runtime-check", "checking runtime", f"Checking {config.runtime.provider} readiness.")
    runtime_ok, runtime_details = run_with_runner(console, runtime_step, setup_steps, engine.healthcheck)

    table = Table()
    table.add_column("setting", style="bold", no_wrap=True)
    table.add_column("value", overflow="fold")
    table.add_column("why", overflow="fold", max_width=42)
    table.add_row("config", str(get_config_path()), _reason("LocalCode stores the local-first runtime and UX defaults here."))
    table.add_row("profile", profile.key, _reason(PROFILE_HINTS.get(profile.key, profile.summary)))
    table.add_row("model", resolved_model, _reason("This is the concrete local model tag LocalCode will try to use."))
    provider_reason = {
        "ollama": "Chosen for the easiest out-of-box local startup path.",
        "mlx-local": "Chosen because Apple Silicon + quantized Gemma is fastest here.",
        "llama_cpp": "Chosen for explicit GGUF tuning and low-latency local serving.",
        "huggingface-local": "Chosen for direct checkpoint control through local Transformers.",
    }.get(config.runtime.provider, "Chosen from the current runtime settings.")
    table.add_row("provider", config.runtime.provider, _reason(provider_reason))
    table.add_row("quant_preset", config.runtime.quant_preset, _reason("Smaller quant settings reduce memory and improve speed on this machine tier."))
    table.add_row("mode", config.runtime.mode, _reason("Fast mode keeps context tighter and reduces startup and generation cost."))
    table.add_row("planner_model", config.runtime.planner_model, _reason("A smaller planner helps route work cheaply before the main model answers."))
    table.add_row("browser", config.browser.mcp_server_name if config.browser.enabled else "disabled", _reason("Browser automation is configured but now loads only when you actually use it."))
    table.add_row("voice", f"{config.voice.stt_provider} + {config.voice.tts_provider}", _reason("Local voice defaults favor offline use and small-machine friendliness."))
    if config.runtime.provider == "ollama":
        table.add_row("ollama_cli", "present" if is_ollama_installed() else "missing", _reason("LocalCode can launch fastest when the local runtime binary is already installed."))
    table.add_row("daemon", "ready" if runtime_ok else "unreachable", _reason("This confirms whether the selected local runtime answered a health check."))
    table.add_row("details", runtime_details, _reason("Concrete backend detail from the runtime check."))
    console.print(table)

    browser_path = ensure_browser_mcp(config) if config.browser.enabled else None
    if browser_path:
        console.print(f"Browser preset saved to {browser_path}")
    for line in browser_status(config):
        console.print(line)

    if auto_install and config.voice.tts_provider == "kokoro":
        ok, details = install_python_packages(["kokoro", "soundfile"])
        console.print(details)
        if not ok:
            return 1
    if auto_install and config.voice.tts_provider == "piper":
        console.print("Skipping automatic Piper install. Piper is the safe Python 3.13 fallback, but it is better installed explicitly with its local model.")
    if auto_install and config.voice.stt_provider == "faster-whisper":
        ok, details = install_python_packages(["faster-whisper"])
        console.print(details)
        if not ok:
            return 1
        pass  # voice removed
        console.print(line)

    if auto_install and config.runtime.provider == "ollama" and is_ollama_installed():
        model_step = SetupStep("model-prepare", "pulling model", f"Preparing local model {resolved_model}.")
        pulled, pull_details = run_with_runner(console, model_step, setup_steps, lambda: pull_model(resolved_model, on_progress=_set_progress))
        console.print(pull_details)
        if not pulled:
            return 1

    if auto_install and config.runtime.provider in {"mlx-local", "huggingface-local"}:
        model_step = SetupStep("model-prepare", "preparing model", f"Resolving local model assets for {resolved_model}.")
        fetched, fetch_details = run_with_runner(
            console,
            model_step,
            setup_steps,
            lambda: fetch_provider_model(config.runtime.provider, config, resolved_model),
        )
        console.print(fetch_details)
        if not fetched:
            return 1

    if config.runtime.provider == "mlx-local":
        if not auto_install:
            console.print("Install MLX local runtime with: pip install -U mlx-lm")
            validate_provider(console, config)
            browser_ok, browser_voice_messages = browser_voice_readiness(config)
            for message in browser_voice_messages:
                console.print(message)
        console.print(f"Set or keep runtime.mlx_model_id = {config.runtime.mlx_model_id or resolved_model}")
        console.print("Then run: gem")
        return 0

    if config.runtime.provider == "huggingface-local":
        if not auto_install:
            console.print("Install local HF runtime with: pip install transformers torch accelerate")
            validate_provider(console, config)
            browser_ok, browser_voice_messages = browser_voice_readiness(config)
            for message in browser_voice_messages:
                console.print(message)
        console.print(f"Set or keep runtime.huggingface_model_id = {config.runtime.huggingface_model_id or resolved_model}")
        console.print("Then run: gem")
        return 0

    if config.runtime.provider == "llama_cpp":
        validate_provider(console, config)
        browser_ok, browser_voice_messages = browser_voice_readiness(config)
        for message in browser_voice_messages:
            console.print(message)
        launch_cmd = runtime_command(config.runtime) or f"{detect_llama_cpp_server_command() or 'llama-server'} -m {resolved_model} --port 8080"
        console.print(f"Start a local llama.cpp server, for example: {launch_cmd}")
        console.print("Or use: gem runtime-up")
        console.print("Then set runtime.base_url to that server, for example http://localhost:8080")
        console.print("Then run: gem")
        return 0

    if not is_ollama_installed():
        plan = detect_install_plan()
        if plan:
            console.print(f"Install Ollama with: {' '.join(plan.command)}")
        else:
            console.print("Install Ollama from https://ollama.com/download")
        console.print(f"Then run: ollama pull {resolved_model}")
        console.print("Then run: gem")
        return 0

    if not runtime_ok:
        launch_cmd = runtime_command(config.runtime) or "ollama serve"
        console.print(f"Start the local runtime with: {launch_cmd}")
        console.print("Or use: gem runtime-up")
    browser_ok, browser_voice_messages = browser_voice_readiness(config)
    for message in browser_voice_messages:
        console.print(message)
    console.print(f"Next: ollama pull {resolved_model}")
    console.print("Then: gem")
    return 0
