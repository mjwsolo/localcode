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

# .browser import removed during T0.9 purge.
from .config import AppConfig, get_config_path, init_config_file, save_config
from .models import ALIASES, GEMMA_PROFILES, MLX_MODEL_IDS, get_runtime_model, resolve_profile
from .performance import MachineProfile, PerformancePreset, apply_preset, benchmark_report, should_promote_legacy_default_to_laptop_26b
from .runtime import LocalCodeRuntimeGateway


def provider_readiness(_runtime_config) -> tuple[bool, list[str]]:
    """Stub: assume provider is ready after provider_checks module removal."""
    return True, []


# browser_voice_readiness removed during T0.9 purge — the browser
# and voice subsystems are gone; no readiness check to stub.


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


def runtime_command(_runtime_config) -> str | None:
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
    """No-op stub kept for one release so external scripts that called
    into bootstrap to "initialise LocalCode" don't suddenly crash with
    AttributeError. The voice stack was purged in T0.9; this function
    can go away after the deprecation window (targets v0.next+2)."""
    return


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
    pkg_dir = Path(__file__).resolve().parent.parent.parent  # src/localcode -> src -> repo root
    candidate = pkg_dir / "llama-cpp-turboquant"
    if (candidate / "CMakeLists.txt").exists():
        return candidate
    # Check home directory
    home_candidate = Path.home() / "llama-cpp-turboquant"
    if (home_candidate / "CMakeLists.txt").exists():
        return home_candidate
    return None


def _turboquant_binary_path() -> Path | None:
    """Return path to the TurboQuant llama-server binary."""
    # Check bundled binary in package first (pip install)
    bundled = Path(__file__).parent / "bin" / "llama-server"
    if bundled.exists():
        return bundled
    # Check data dir (downloaded by bootstrap)
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
    import ssl
    import urllib.request
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine == "arm64":
        plat = "macos-arm64"
    elif system == "linux" and machine in ("x86_64", "amd64"):
        plat = "linux-x86_64"
    else:
        return False, f"No pre-built binary for {system}-{machine}. Clone the repo and build from source."

    version = "0.1.8"
    url = _BINARY_RELEASE_URL.format(version=version, platform=plat)
    data_dir = Path.home() / ".local" / "share" / "localcode"
    data_dir.mkdir(parents=True, exist_ok=True)
    binary = data_dir / "llama-server"

    if on_progress:
        on_progress(f"downloading llama-server for {plat}...")
    try:
        try:
            import certifi
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_ctx = ssl.create_default_context()
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, str(binary))
        binary.chmod(0o755)
        return True, str(binary)
    except ssl.SSLCertVerificationError:
        # Retry without verification as last resort
        ssl_ctx = ssl._create_unverified_context()
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, str(binary))
        binary.chmod(0o755)
        return True, str(binary)
    except Exception as e:
        return False, f"Download failed: {e}\nBuild from source instead: clone llama-cpp-turboquant and run cmake."


def _is_complete_download(p: Path, catalog_entry) -> bool:
    """Cheap completeness check: file must exist AND be within 1% of the
    catalog's declared `size_gb`. A partial download from a failed attempt
    will be far smaller (or in rare GGUF tail-buffer cases, slightly off).

    If we have no catalog entry to compare against (user dropped a manual
    GGUF), trust the file as long as it's at least 1 MiB — better than
    rejecting legitimate user files.
    """
    try:
        size = p.stat().st_size
    except OSError:
        return False
    if catalog_entry is None:
        return size > 1024 * 1024
    # Catalog `size_gb` is DECIMAL GB (HF's convention, e.g. 11.2 GB =
    # 11.2e9 bytes), so compare in 1000**3 — NOT 1024**3. Using binary
    # GiB here over-estimated the expected bytes by ~7% and wrongly
    # rejected complete files (a real 11.29e9-byte model "failed" an
    # expected-12.0e9 check), so the server never started.
    expected = int(catalog_entry.size_gb * 1000 ** 3)
    # Allow 3% tolerance for rounding in the catalog's GB figure.
    return size >= int(expected * 0.97)


def get_model_path(preferred_filename: str | None = None) -> Path | None:
    """Return path to a usable GGUF model, checking multiple locations.

    If `preferred_filename` is given, only files matching that exact name are
    returned — the caller picks the model, this just locates it on disk. If
    None, falls back to whatever is set in config or the first file in the
    canonical models dir.

    Prioritizes Ollama blob (integrity-verified by Ollama) over HF downloads.
    """
    from .models_catalog import CHOICES, model_dir as _model_dir

    # If we have a preferred filename but it's not in the catalog, we still
    # honor it (user might have manually dropped a GGUF in the models dir).
    catalog_entry = None
    if preferred_filename is not None:
        for c in CHOICES:
            if c.filename == preferred_filename:
                catalog_entry = c
                break

    # 1. Ollama blob — only if the catalog entry has a documented tag. The
    # original hardcoded `gemma26b-iq3` tag is specific to one user's Ollama
    # install; we don't assume anyone else has named it that way.
    #    (Kept here as a comment because Ollama integrity-verifies the blob —
    #    if a caller wants it back, thread the tag through ModelChoice.)

    # 2. Config model path (wins if file exists, matches preferred filename,
    # AND is the right size — a partial download from a failed attempt is
    # NOT a usable model).
    from .config import load_config
    try:
        config = load_config()
        if config.runtime.model and Path(config.runtime.model).is_file():
            p = Path(config.runtime.model)
            if (preferred_filename is None or p.name == preferred_filename) \
                    and _is_complete_download(p, catalog_entry):
                return p
    except Exception:
        pass

    # 3. Canonical download directory (same size check — a half-written file
    # in the canonical dir is a partial, not a usable model).
    if preferred_filename is not None:
        candidate = _model_dir() / preferred_filename
        if candidate.is_file() and _is_complete_download(candidate, catalog_entry):
            return candidate
    else:
        # No preference — pick any catalog model that's already downloaded,
        # preferring the first in CHOICES order.
        for c in CHOICES:
            if c.local_path.is_file():
                return c.local_path

    # 4. Common locations (only if no filename preference)
    if preferred_filename is None:
        for search_dir in [Path.home() / "models", Path.home() / ".cache" / "huggingface"]:
            if search_dir.is_dir():
                for f in search_dir.rglob("*.gguf"):
                    # Match against any catalog filename
                    if any(c.filename == f.name for c in CHOICES):
                        return f

    return None


def _download_parallel(url: str, dest: Path, num_threads: int = 16,
                       on_progress: Callable[[str], None] | None = None) -> None:
    """Download a large file using parallel HTTP range requests.

    Uses 16 threads and 1MB buffers for maximum throughput.
    Falls back to single-threaded if the server doesn't support ranges.
    """
    import ssl
    import threading
    import urllib.request

    # Create SSL context — try certifi, then system, then unverified
    ssl_ctx = None
    for _attempt in range(3):
        try:
            if _attempt == 0:
                import certifi
                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            elif _attempt == 1:
                ssl_ctx = ssl.create_default_context()
            else:
                ssl_ctx = ssl._create_unverified_context()
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, context=ssl_ctx, timeout=10).close()
            break  # this context works
        except ImportError:
            continue
        except Exception:
            continue
    if ssl_ctx is None:
        ssl_ctx = ssl._create_unverified_context()

    # Get file size via HEAD
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, context=ssl_ctx) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))
        accepts_ranges = resp.headers.get("Accept-Ranges", "none") != "none"

    if not total_size or not accepts_ranges:
        # Fallback: single-threaded
        def _report(block_num, block_size, _total):
            if on_progress and total_size > 0:
                done = min(block_num * block_size, total_size)
                on_progress(f"Downloading: {done // (1024*1024)}/{total_size // (1024*1024)} MB ({done * 100 // total_size}%)")
        urllib.request.urlretrieve(url, str(dest), reporthook=_report)  # no ssl_ctx for urlretrieve
        return

    # Split into chunks
    chunk_size = total_size // num_threads
    downloaded = [0] * num_threads  # bytes per thread for progress
    errors: list[str] = []
    lock = threading.Lock()

    # Pre-allocate the output file
    with open(dest, "wb") as f:
        f.seek(total_size - 1)
        f.write(b"\0")

    def _download_chunk(idx: int, start: int, end: int) -> None:
        try:
            req = urllib.request.Request(url)
            req.add_header("Range", f"bytes={start}-{end}")
            with urllib.request.urlopen(req, context=ssl_ctx) as resp:
                buf_size = 8 * 1024 * 1024  # 8MB read buffer — cuts syscall overhead vs 1MB
                with open(dest, "r+b") as f:
                    f.seek(start)
                    while True:
                        data = resp.read(buf_size)
                        if not data:
                            break
                        f.write(data)
                        with lock:
                            downloaded[idx] += len(data)
        except Exception as e:
            with lock:
                errors.append(f"Chunk {idx}: {e}")

    # Progress reporter
    progress_stop = threading.Event()

    def _report_progress() -> None:
        while not progress_stop.is_set():
            total_done = sum(downloaded)
            mb_done = total_done // (1024 * 1024)
            mb_total = total_size // (1024 * 1024)
            pct = min(100, total_done * 100 // total_size) if total_size else 0
            speed_label = ""
            if on_progress:
                on_progress(f"Downloading: {mb_done}/{mb_total} MB ({pct}%){speed_label}")
            progress_stop.wait(0.5)

    progress_thread = threading.Thread(target=_report_progress, daemon=True)
    progress_thread.start()

    # Launch parallel downloads
    threads: list[threading.Thread] = []
    for i in range(num_threads):
        start = i * chunk_size
        end = total_size - 1 if i == num_threads - 1 else (i + 1) * chunk_size - 1
        t = threading.Thread(target=_download_chunk, args=(i, start, end))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    progress_stop.set()
    progress_thread.join(timeout=1)

    if errors:
        dest.unlink(missing_ok=True)
        raise RuntimeError("; ".join(errors))


def download_model(
    choice=None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Download a GGUF model selected from `models_catalog.CHOICES`.

    If `choice` is None, defaults to the first entry in CHOICES (Gemma 4 26B).
    Returns (success, path_or_error_message).
    """
    from .models_catalog import CHOICES

    if choice is None:
        choice = CHOICES[0]

    # Already present?
    existing = get_model_path(choice.filename)
    if existing:
        return True, str(existing)

    # Direct HuggingFace download (parallel, resumable via our own implementation).
    # We don't attempt Ollama pull here because Ollama's named tags don't map
    # 1:1 to unsloth's GGUF filenames — the catalog is the single source of truth.
    model_dir = choice.local_path.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = choice.local_path

    if model_file.exists():
        return True, str(model_file)

    url = f"https://huggingface.co/{choice.hf_repo}/resolve/main/{choice.filename}"
    if on_progress:
        on_progress(f"Downloading {choice.filename} (~{choice.size_gb:.1f} GB)...")

    # Retry with exponential backoff. Both hf_transfer and urllib paths
    # support resume — we DO NOT delete the partial file between attempts
    # so retries pick up where the last attempt left off.
    last_err: Exception | None = None
    last_err_category = "unknown"
    backoffs = [0, 2, 5]  # seconds; 3 attempts total
    for attempt, delay in enumerate(backoffs, start=1):
        if delay:
            if on_progress:
                on_progress(f"Retry {attempt}/{len(backoffs)} in {delay}s...")
            import time as _t
            _t.sleep(delay)
        # Fast path: hf_transfer (Rust). Resumes via huggingface_hub's
        # built-in `.incomplete` partial file handling.
        try:
            if _try_hf_transfer_download(choice, model_file, on_progress):
                return True, str(model_file)
        except Exception as e:
            last_err = e
            last_err_category = _classify_download_error(e)
            if last_err_category in ("disk_full", "auth", "not_found"):
                # Non-retryable — fail fast so the UI can show the right message.
                return False, _format_download_error(last_err_category, e)
            if on_progress:
                on_progress(
                    f"hf_transfer failed ({last_err_category}); trying urllib fallback"
                )
        # Slow path: tuned urllib parallel downloader.
        try:
            _download_parallel(url, model_file, num_threads=32, on_progress=on_progress)
            return True, str(model_file)
        except Exception as e:
            last_err = e
            last_err_category = _classify_download_error(e)
            if last_err_category in ("disk_full", "auth", "not_found"):
                return False, _format_download_error(last_err_category, e)
            # Keep the partial file for the next retry attempt.

    # All retries exhausted. Leave the partial in place — user can restart
    # localcode and the next download attempt will resume from where this
    # one left off.
    return False, _format_download_error(
        last_err_category,
        last_err or RuntimeError("unknown download error"),
    )


def _classify_download_error(e: Exception) -> str:
    """Map a download exception to a stable category for UI + retry policy."""
    msg = (str(e) or "").lower()
    et = type(e).__name__
    if "no space" in msg or "disk full" in msg or "enospc" in msg or et == "OSError" and "28" in msg:
        return "disk_full"
    if "401" in msg or "403" in msg or "gated" in msg or "permission" in msg:
        return "auth"
    if "404" in msg or "not found" in msg:
        return "not_found"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "connection" in msg or "network" in msg or "dns" in msg or "resolve" in msg:
        return "network"
    if "ssl" in msg or "certificate" in msg:
        return "ssl"
    return "unknown"


def _format_download_error(category: str, e: Exception) -> str:
    """User-facing error string for a classified download failure."""
    raw = str(e)[:200].replace("\n", " ").strip()
    by_cat = {
        "disk_full":  f"Not enough disk space to finish the download. Free up space and retry.\n[{raw}]",
        "auth":       f"Hugging Face rejected the request — model may be gated or require a token.\n[{raw}]",
        "not_found":  f"Model file not found on Hugging Face — the catalog entry's filename may be stale.\n[{raw}]",
        "timeout":    f"Download timed out. Check your connection and retry.\n[{raw}]",
        "network":    f"Network error during download. Check your connection and retry.\n[{raw}]",
        "ssl":        f"SSL/TLS error during download — your system trust store may be out of date.\n[{raw}]",
        "unknown":    f"Model download failed: {raw}",
    }
    return by_cat.get(category, by_cat["unknown"])


def download_mmproj(
    choice,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Download the vision projector (mmproj.gguf) for a catalog entry.

    Returns (ok, path_or_error). Idempotent — short-circuits if the file
    is already present. Reuses download_model's retry/backoff/fast-path
    machinery via a proxy catalog entry that points at the mmproj file.
    """
    if not choice.supports_vision or not choice.mmproj_filename:
        return False, "Model has no vision projector defined."
    dest = choice.mmproj_path
    if dest is None:
        return False, "mmproj_path resolved to None."
    if dest.is_file() and _is_mmproj_complete(dest, choice):
        return True, str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    from dataclasses import replace
    # `mmproj_hf_filename` is what's actually in the HF repo (often
    # just "mmproj-F16.gguf" — shared across model families). We download
    # under THAT name, then rename to `mmproj_filename` (which is unique
    # per model family) so the Gemma and Qwen projectors don't overwrite
    # each other in the shared model_dir.
    hf_name = choice.mmproj_hf_filename or choice.mmproj_filename
    proxy = replace(
        choice,
        filename=hf_name,
        size_gb=choice.mmproj_size_gb or 1.0,
        mmproj_filename=None,
        mmproj_size_gb=0.0,
        mmproj_hf_filename=None,
    )
    ok, result = download_model(proxy, on_progress=on_progress)
    if not ok:
        return ok, result
    # Rename to the catalog's local filename if it differs from the HF name.
    from pathlib import Path as _P
    downloaded = _P(result)
    if downloaded.name != choice.mmproj_filename:
        target = dest  # choice.mmproj_path resolved earlier
        try:
            if target.exists():
                target.unlink()
            downloaded.rename(target)
            return True, str(target)
        except Exception as e:
            return False, f"Couldn't rename mmproj to {target.name}: {e}"
    return ok, result


def _is_mmproj_complete(p: Path, choice) -> bool:
    """Cheap completeness check for the mmproj sidecar — like
    `_is_complete_download` but with 10% slack since catalog mmproj
    size estimates can be slightly off."""
    try:
        size = p.stat().st_size
    except OSError:
        return False
    if not choice.mmproj_size_gb:
        return size > 10 * 1024 * 1024  # >10 MB is presumably real
    expected = int(choice.mmproj_size_gb * 1024 ** 3)
    return size >= int(expected * 0.9)


def _try_hf_transfer_download(choice, dest: Path, on_progress: Callable[[str], None] | None) -> bool:
    """Try to download via huggingface_hub + hf_transfer (Rust-backed).

    Returns True on success, raises on hard failure. Returns False if the
    library is not available (caller should fall back).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    # hf_transfer is opt-in via env var; huggingface_hub auto-uses it when
    # available and HF_HUB_ENABLE_HF_TRANSFER=1.
    try:
        import hf_transfer  # noqa: F401  -- presence check
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        # huggingface_hub alone is still faster than urllib because it does
        # connection reuse + better chunking, so we still use it without
        # the Rust extension.
        pass
    if on_progress:
        on_progress(f"Downloading {choice.filename} via hf_transfer (Rust)...")
    downloaded_path = hf_hub_download(
        repo_id=choice.hf_repo,
        filename=choice.filename,
        local_dir=str(dest.parent),
        local_dir_use_symlinks=False,
    )
    # huggingface_hub may write to a slightly different filename inside
    # local_dir; symlink/rename to our expected dest.
    dp = Path(downloaded_path)
    if dp.resolve() != dest.resolve():
        try:
            if dest.exists():
                dest.unlink()
            dp.rename(dest)
        except Exception:
            # Last resort: copy
            import shutil as _sh
            _sh.copyfile(dp, dest)
    return dest.is_file()


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
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
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


def choose_model_for_setup(console: Console, config: AppConfig, assume_defaults: bool = False):
    """Interactive picker for the GGUF model to download during setup.

    Returns the chosen `ModelChoice`. In `assume_defaults` mode (or non-TTY),
    picks whichever catalog entry matches the already-configured model file,
    otherwise the first entry.
    """
    from .models_catalog import CHOICES, current as current_choice, by_filename

    cfg_current = current_choice(config)

    # Non-interactive path: honor config, else first entry.
    if assume_defaults or not sys.stdin.isatty():
        return cfg_current or CHOICES[0]

    default_idx = 0
    if cfg_current is not None:
        for i, c in enumerate(CHOICES):
            if c.key == cfg_current.key:
                default_idx = i
                break

    console.print()
    console.print("[bold]Choose a local model to download:[/]")
    console.print()
    for index, c in enumerate(CHOICES, start=1):
        downloaded = c.local_path.is_file()
        status = "[green]downloaded[/]" if downloaded else f"will download {c.size_gb:.1f} GB"
        marker = "  (current)" if cfg_current and cfg_current.key == c.key else ""
        swe = (
            f"HumanEval {c.humaneval_pass_at_1*100:.1f}%"
            if c.humaneval_pass_at_1 is not None else "no benchmark"
        )
        console.print(f"  [bold]{index}.[/] {c.name}{marker}")
        console.print(f"     active:   {c.active_params}  |  arch: {c.architecture}  |  {swe}  |  {c.license}")
        console.print(f"     size:     {c.size_gb:.1f} GB   ({status})")
        console.print(f"     source:   https://huggingface.co/{c.hf_repo}")
        console.print(f"     file:     {c.filename}")
        console.print(f"     saves to: {c.local_path}")
        console.print(f"     note:     {c.notes}")
        console.print()
    raw = input(f"Select [1-{len(CHOICES)}] or press Enter for {CHOICES[default_idx].name}: ").strip()
    if not raw:
        return CHOICES[default_idx]
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(CHOICES):
            return CHOICES[idx - 1]
    # Accept short key too
    for c in CHOICES:
        if c.key == raw.lower():
            return c
    # Accept filename match too (user paste-friendly)
    by_name = by_filename(raw)
    if by_name is not None:
        return by_name
    console.print(f"[yellow]Unknown choice '{raw}', using {CHOICES[default_idx].name}.[/]")
    return CHOICES[default_idx]


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

    # Pick which GGUF to use (catalog-driven, interactive unless assume_defaults)
    # and download it if not already present.
    if auto_install and config.runtime.provider == "llama_cpp":
        chosen = choose_model_for_setup(console, config, assume_defaults=assume_defaults)
        model_path = get_model_path(chosen.filename)
        if model_path:
            console.print(f"Model already downloaded: {model_path}")
        else:
            model_step = SetupStep("model-prepare", "downloading model", f"Downloading {chosen.name} (~{chosen.size_gb:.1f} GB).")
            ok, model_result = run_with_runner(
                console, model_step, setup_steps,
                lambda: download_model(chosen, on_progress=_set_progress),
            )
            if ok:
                model_path = Path(model_result)
                console.print(f"Model downloaded: {model_path}")
            else:
                console.print(f"Model download failed: {model_result}")
                return 1
        # Persist the picked model to config so runtime.py + /model command see it
        if model_path:
            config.runtime.model = str(model_path)
        # Ensure base_url is correct for llama_cpp
        if "8081" not in config.runtime.base_url:
            config.runtime.base_url = "http://localhost:8081"
        save_config(config)

    engine = LocalCodeRuntimeGateway(config.runtime)
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
    # browser / voice rows removed during T0.9 purge.
    if config.runtime.provider == "ollama":
        table.add_row("ollama_cli", "present" if is_ollama_installed() else "missing", _reason("LocalCode can launch fastest when the local runtime binary is already installed."))
    table.add_row("daemon", "ready" if runtime_ok else "unreachable", _reason("This confirms whether the selected local runtime answered a health check."))
    table.add_row("details", runtime_details, _reason("Concrete backend detail from the runtime check."))
    console.print(table)

    # Browser MCP preset + voice installs removed during T0.9 purge.

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
            # browser_voice_readiness calls removed during T0.9 purge.
        console.print(f"Set or keep runtime.mlx_model_id = {config.runtime.mlx_model_id or resolved_model}")
        console.print("Then run: localcode")
        return 0

    if config.runtime.provider == "huggingface-local":
        if not auto_install:
            console.print("Install local HF runtime with: pip install transformers torch accelerate")
            validate_provider(console, config)
            # browser_voice_readiness calls removed during T0.9 purge.
        console.print(f"Set or keep runtime.huggingface_model_id = {config.runtime.huggingface_model_id or resolved_model}")
        console.print("Then run: localcode")
        return 0

    if config.runtime.provider == "llama_cpp":
        validate_provider(console, config)
        browser_ok, browser_voice_messages = browser_voice_readiness(config)
        for message in browser_voice_messages:
            console.print(message)
        launch_cmd = runtime_command(config.runtime) or f"{detect_llama_cpp_server_command() or 'llama-server'} -m {resolved_model} --port 8080"
        console.print(f"Start a local llama.cpp server, for example: {launch_cmd}")
        console.print("Or use: localcode runtime-up")
        console.print("Then set runtime.base_url to that server, for example http://localhost:8080")
        console.print("Then run: localcode")
        return 0

    if not is_ollama_installed():
        plan = detect_install_plan()
        if plan:
            console.print(f"Install Ollama with: {' '.join(plan.command)}")
        else:
            console.print("Install Ollama from https://ollama.com/download")
        console.print(f"Then run: ollama pull {resolved_model}")
        console.print("Then run: localcode")
        return 0

    if not runtime_ok:
        launch_cmd = runtime_command(config.runtime) or "ollama serve"
        console.print(f"Start the local runtime with: {launch_cmd}")
        console.print("Or use: localcode runtime-up")
    # browser_voice_readiness call removed during T0.9 purge.
    console.print(f"Next: ollama pull {resolved_model}")
    console.print("Then: localcode")
    return 0
