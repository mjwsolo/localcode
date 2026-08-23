from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable, TypeVar

# huggingface_hub snapshots these env vars into `huggingface_hub.constants`
# AT IMPORT TIME — setting them after the first `import huggingface_hub`
# anywhere in the process silently does nothing. Set them here, before any
# code path can import the hub:
#   HF_HUB_DISABLE_XET — force the hub's plain-HTTP downloader instead of
#     the hf_xet backend. Xet is faster on a clean run, but it CANNOT
#     resume an interrupted download: it rewrites the `.incomplete` file
#     from its chunk cache, and on a default install that cache holds
#     ~nothing — so quitting LocalCode mid-download restarted a 33 GB
#     model from byte 0 (observed 2026-06-11). The HTTP path resumes via
#     `.incomplete` + Range. For 10-35 GB GGUFs on desktop machines,
#     surviving an interrupt is worth more than peak throughput.
#   HF_HUB_DISABLE_PROGRESS_BARS — keeps tqdm off stderr, which would
#     corrupt the Textual screen; progress reaches the TUI through the
#     `tqdm_class` hook in `_try_hub_download` instead.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── Background model-download manager ────────────────────────────────
#
# Guards _DOWNLOADS and _DOWNLOAD_QUEUE together. Every read/mutation of
# either structure happens inside this lock. Callers NEVER hold the lock
# while doing I/O — copies are returned, not live references.
_DOWNLOAD_LOCK = threading.Lock()

# model_key -> status dict (the EXACT shape in the contract).
_DOWNLOADS: dict[str, dict] = {}

# model_keys waiting for a free slot, FIFO. Each also has a "queued"
# entry in _DOWNLOADS.
_DOWNLOAD_QUEUE: list[str] = []

# model_key -> choice, captured at enqueue so the worker still has its
# ModelChoice after waiting in the queue.
_DOWNLOAD_CHOICES: dict[str, object] = {}

# Max concurrent ACTIVE downloads (status == "downloading"). The rest sit
# in _DOWNLOAD_QUEUE with status "queued" until a slot frees.
_MAX_ACTIVE_DOWNLOADS = 2

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# .browser import removed during T0.9 purge.
from .config import AppConfig, get_config_path, init_config_file, save_config
from .models import ALIASES, GEMMA_PROFILES, get_runtime_model, resolve_profile
from .performance import MachineProfile, PerformancePreset, apply_preset, benchmark_report, should_promote_legacy_default_to_laptop_26b
from .runtime import LocalCodeRuntimeGateway


@dataclass(frozen=True)
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

# Generic return type for run_with_runner (the TypeVar was imported but never
# bound — ruff F821, and it broke the E9/F82 lint gate in CI).
T = TypeVar("T")


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


def _bundled_binary_path(name: str) -> Path | None:
    """Resolve a shipped llama.cpp binary by name, in fixed order:

    1. the copy bundled inside the package (`src/localcode/bin/`, what the
       wheel ships);
    2. the user data dir (`~/.local/share/localcode/`, a manual drop-in);
    3. a developer source build (`llama-cpp-turboquant/build/bin/`).

    Nothing here downloads, clones or compiles. If the binary is not on
    disk the answer is None and the caller reports a broken install.
    """
    bundled = Path(__file__).parent / "bin" / name
    if bundled.exists():
        return bundled
    data_binary = Path.home() / ".local" / "share" / "localcode" / name
    if data_binary.exists():
        return data_binary
    source = _find_turboquant_source()
    if source is None:
        return None
    binary = source / "build" / "bin" / name
    if binary.exists():
        return binary
    return None


def _turboquant_binary_path() -> Path | None:
    """Return path to the bundled llama-server binary."""
    return _bundled_binary_path("llama-server")


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
        # SECURITY: never fetch an EXECUTABLE over an unverified connection — a
        # network attacker could swap in a malicious binary. If cert
        # verification fails, the fix is to repair the trust store, not to
        # accept any cert.
        return False, (
            "Download failed: TLS certificate verification failed. Update your "
            "certificates (`pip install -U certifi`), or build llama-cpp-"
            "turboquant from source. LocalCode will not download an executable "
            "over an unverified connection."
        )
    except Exception as e:
        return False, f"Download failed: {e}\nReinstall localcode; the server ships inside the package."


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


def _verify_download_integrity(choice, path: Path) -> tuple[bool, str]:
    """Verify a freshly-downloaded model file against its pinned sha256.

    Returns (ok, reason). When the catalog entry has no `sha256` set (the
    default), this is a no-op that returns (True, "") — behavior is unchanged
    for unpinned entries. When a hash IS pinned and the file does not match,
    the file is DELETED (so a poisoned artifact is never loaded or executed)
    and (False, reason) is returned.

    This is the integrity backstop the size-only completeness check is not:
    upstream can swap weights under the same filename, and TLS/host pinning
    alone does not detect a malicious or corrupted artifact served by a
    compromised upstream repo.
    """
    expected = getattr(choice, "sha256", None)
    if not expected:
        return True, ""
    try:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        actual = h.hexdigest()
    except Exception as e:
        return False, f"Integrity check could not read {path.name}: {e}"
    if actual.lower() != str(expected).lower():
        try:
            path.unlink()
        except Exception:
            pass
        try:
            from .events import emit as _emit
            _emit("download_integrity_fail", filename=choice.filename, expected=str(expected)[:16], actual=actual[:16])
        except Exception:
            pass
        return False, (
            f"Integrity check FAILED for {choice.filename}: expected sha256 "
            f"{expected}, got {actual}. The file was deleted — LocalCode will "
            "not load a model whose contents don't match the pinned hash."
        )
    return True, ""


def get_model_path(preferred_filename: str | None = None) -> Path | None:
    """Return path to a usable GGUF model, checking multiple locations.

    If `preferred_filename` is given, only files matching that exact name are
    returned — the caller picks the model, this just locates it on disk. If
    None, falls back to whatever is set in config or the first file in the
    canonical models dir.
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

    # 1. Config model path (wins if file exists, matches preferred filename,
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

    Falls back to single-threaded if the server doesn't support ranges.

    All bytes land in `<dest>.part`, atomically renamed to `dest` only on
    success. NEVER write to the final name directly: the parallel path
    pre-allocates the full file size up front, so a kill mid-download
    used to leave a full-size mostly-zeros file at the final name that
    passed the size-based completeness check and then failed inside
    llama-server with a cryptic GGUF error.
    """
    import ssl
    import threading
    import urllib.request

    # Create SSL context — try certifi, then system, then unverified.
    # The probing HEAD doubles as the size/range capability check, so the
    # common case (certifi works) costs exactly ONE round-trip before data
    # flows instead of two.
    ssl_ctx = None
    total_size = 0
    accepts_ranges = False
    for _attempt in range(3):
        try:
            if _attempt == 0:
                import certifi
                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            else:
                # SECURITY: stay verified — no unverified-TLS fallback. If the
                # system trust store can't verify, the download fails cleanly
                # rather than trusting any cert.
                ssl_ctx = ssl.create_default_context()
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=10) as resp:
                total_size = int(resp.headers.get("Content-Length", 0))
                accepts_ranges = resp.headers.get("Accept-Ranges", "none") != "none"
            break  # this context works
        except ImportError:
            continue
        except Exception:
            continue
    if ssl_ctx is None:
        # SECURITY: verified context only — never fall back to unverified TLS.
        ssl_ctx = ssl.create_default_context()

    part = dest.with_name(dest.name + ".part")

    if not total_size or not accepts_ranges:
        # Fallback: single-threaded
        def _report(block_num, block_size, _total):
            if on_progress and total_size > 0:
                done = min(block_num * block_size, total_size)
                on_progress(f"Downloading: {done // (1024*1024)}/{total_size // (1024*1024)} MB ({done * 100 // total_size}%)")
        urllib.request.urlretrieve(url, str(part), reporthook=_report)  # no ssl_ctx for urlretrieve
        part.replace(dest)
        return

    # Split into chunks
    chunk_size = total_size // num_threads
    downloaded = [0] * num_threads  # bytes per thread for progress
    errors: list[str] = []
    lock = threading.Lock()

    # Pre-allocate the working file
    with open(part, "wb") as f:
        f.seek(total_size - 1)
        f.write(b"\0")

    def _download_chunk(idx: int, start: int, end: int) -> None:
        try:
            req = urllib.request.Request(url)
            req.add_header("Range", f"bytes={start}-{end}")
            # timeout bounds CONNECT + each blocking read, not the whole
            # transfer — a stalled thread aborts in 60s instead of hanging
            # the download forever.
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as resp:
                buf_size = 8 * 1024 * 1024  # 8MB read buffer — cuts syscall overhead vs 1MB
                with open(part, "r+b") as f:
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
        part.unlink(missing_ok=True)
        raise RuntimeError("; ".join(errors))
    # All chunks verified-written — promote atomically to the final name.
    part.replace(dest)


def download_model(
    choice=None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Download a GGUF model selected from `models_catalog.CHOICES`.

    If `choice` is None, falls back to `recommend()` — the RAM-appropriate model
    for this machine, not a hardcoded default.
    Returns (success, path_or_error_message).
    """
    from .models_catalog import recommend

    if choice is None:
        # No explicit choice → the RAM-appropriate model for THIS machine,
        # not a hardcoded default.
        choice = recommend()

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

    # A file at the final name only counts when it's the full size — a
    # partial from an interrupted attempt must fall through to the
    # download (the hub path resumes; treating the partial as "done"
    # produced a model llama-server couldn't load).
    if model_file.exists() and _is_complete_download(model_file, choice):
        return True, str(model_file)

    # Disk-space preflight. A 33 GB download that fills the disk fails
    # cryptically near the end (and leaves a useless partial); catching
    # it up front with the real numbers is far kinder. Require headroom
    # for the bytes still needed (size minus any partial already on disk)
    # plus a 2 GB safety margin so we don't wedge the whole system.
    try:
        import shutil as _shutil
        already = model_file.stat().st_size if model_file.exists() else 0
        need = max(0, int(choice.size_gb * 1000 ** 3) - already)
        free = _shutil.disk_usage(model_dir).free
        if free < need + 2 * 1024 ** 3:
            need_gb = need / 1000 ** 3
            free_gb = free / 1000 ** 3
            return False, (
                f"Not enough disk space: {choice.name} needs ~{need_gb:.1f} GB "
                f"but only {free_gb:.1f} GB is free in {model_dir}. Free up space "
                f"or pick a smaller quant in /model."
            )
    except Exception:
        # Never let the preflight itself block a download — if disk_usage
        # fails (exotic FS), fall through and let the download try.
        pass

    _rev = getattr(choice, "revision", None) or "main"
    url = f"https://huggingface.co/{choice.hf_repo}/resolve/{_rev}/{choice.filename}"
    if on_progress:
        on_progress(f"Downloading {choice.filename} (~{choice.size_gb:.1f} GB)...")

    # Retry with exponential backoff. The hub path resumes from its
    # `.incomplete` sidecar (kept across attempts AND across app
    # restarts); the urllib fallback restarts its own `.part` file but
    # never touches the hub's partial, so a later hub attempt still
    # resumes.
    last_err: Exception | None = None
    last_err_category = "unknown"
    backoffs = [0, 2, 5]  # seconds; 3 attempts total
    for attempt, delay in enumerate(backoffs, start=1):
        if delay:
            if on_progress:
                on_progress(f"Retry {attempt}/{len(backoffs)} in {delay}s...")
            import time as _t
            _t.sleep(delay)
        # Fast path: huggingface_hub (hf_xet Rust backend on Xet repos).
        # Resumes via the hub's built-in partial-download handling.
        try:
            if _try_hub_download(choice, model_file, on_progress):
                ok, reason = _verify_download_integrity(choice, model_file)
                if not ok:
                    return False, reason
                return True, str(model_file)
        except Exception as e:
            last_err = e
            last_err_category = _classify_download_error(e)
            if last_err_category in ("disk_full", "auth", "not_found"):
                # Non-retryable — fail fast so the UI can show the right message.
                return False, _format_download_error(last_err_category, e)
            if on_progress:
                on_progress(
                    f"Hub download failed ({last_err_category}); trying urllib fallback"
                )
        # Slow path: tuned urllib parallel downloader.
        try:
            _download_parallel(url, model_file, num_threads=32, on_progress=on_progress)
            ok, reason = _verify_download_integrity(choice, model_file)
            if not ok:
                return False, reason
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


# ── Background model-download public API ─────────────────────────────
#
# Built ON TOP of download_model() — its resume / disk-preflight /
# retry / progress behaviour is reused unchanged. These functions only
# manage a thread-safe registry of in-flight downloads and a 2-slot
# scheduler; the actual byte-moving is still download_model().


def model_key_for(choice) -> str:
    """Stable registry key for a catalog choice — its unique filename."""
    return choice.filename


def is_download_complete(choice) -> bool:
    """True iff a fully-downloaded, correctly-sized file exists on disk.

    Reuses get_model_path's existing completeness logic (which runs
    _is_complete_download against the catalog entry) — no new size math.
    """
    return get_model_path(choice.filename) is not None


def _parse_progress(line: str) -> tuple[int | None, int | None, int | None]:
    """Parse a download_model progress line into (downloaded_mb, total_mb, pct).

    Recognises the `_make_progress_tqdm` / `_download_parallel` format:
        "Downloading: <done>/<total> MB (<pct>%)"
    Any field that fails to parse is returned as None (callers leave the
    corresponding registry field unchanged). Non-MB lines
    ("Downloading…", "Retry 1/3 in 2s...") parse to (None, None, None).
    """
    if not isinstance(line, str):
        return (None, None, None)
    s = line.strip()
    if "MB" not in s or "/" not in s:
        return (None, None, None)
    downloaded_mb: int | None = None
    total_mb: int | None = None
    pct: int | None = None
    # Carve out the "<done>/<total> MB" segment.
    try:
        before_mb = s.split("MB", 1)[0]  # e.g. "Downloading: 1200/11200 "
        frac = before_mb.split(":", 1)[-1].strip()  # "1200/11200"
        if "/" in frac:
            d_str, t_str = frac.split("/", 1)
            d_str, t_str = d_str.strip(), t_str.strip()
            if d_str.isdigit():
                downloaded_mb = int(d_str)
            if t_str.isdigit():
                total_mb = int(t_str)
    except Exception:
        pass
    # Carve out "(<pct>%)".
    try:
        if "(" in s and "%" in s:
            pct_str = s.split("(", 1)[1].split("%", 1)[0].strip()
            if pct_str.isdigit():
                pct = int(pct_str)
    except Exception:
        pass
    return (downloaded_mb, total_mb, pct)


def _apply_progress(key: str, line: str) -> None:
    """Parse a progress line and update the registry entry under the lock."""
    downloaded_mb, total_mb, pct = _parse_progress(line)
    if downloaded_mb is None and total_mb is None and pct is None:
        return
    with _DOWNLOAD_LOCK:
        entry = _DOWNLOADS.get(key)
        if entry is None:
            return
        if downloaded_mb is not None:
            entry["downloaded_mb"] = downloaded_mb
        if total_mb is not None:
            entry["total_mb"] = total_mb
        if pct is not None:
            entry["progress_pct"] = pct


def _maybe_start_next() -> None:
    """Promote queued downloads into running threads up to the slot cap.

    MUST be called while holding _DOWNLOAD_LOCK.
    """
    active = sum(1 for e in _DOWNLOADS.values() if e["status"] == "downloading")
    while active < _MAX_ACTIVE_DOWNLOADS and _DOWNLOAD_QUEUE:
        key = _DOWNLOAD_QUEUE.pop(0)
        entry = _DOWNLOADS.get(key)
        choice = _DOWNLOAD_CHOICES.get(key)
        if entry is None or choice is None:
            continue
        entry["status"] = "downloading"
        threading.Thread(
            target=_run_download, args=(key, choice), daemon=True
        ).start()
        active += 1


def _run_download(key: str, choice) -> None:
    """Daemon worker — runs download_model OUTSIDE the lock during I/O."""
    on_progress = lambda line: _apply_progress(key, line)
    try:
        ok, result = download_model(choice, on_progress=on_progress)
    except Exception as exc:  # pragma: no cover - defensive; download_model traps its own
        ok, result = False, str(exc)
    try:
        with _DOWNLOAD_LOCK:
            entry = _DOWNLOADS.get(key)
            if entry is not None:
                if ok:
                    entry["status"] = "done"
                    entry["progress_pct"] = 100
                    entry["error"] = None
                else:
                    entry["status"] = "failed"
                    entry["error"] = result
    finally:
        with _DOWNLOAD_LOCK:
            _maybe_start_next()


def start_background_download(choice) -> str:
    """Enqueue a background download for `choice`; return its model_key.

    Returns immediately. The actual download runs on a daemon thread via
    download_model(), respecting the _MAX_ACTIVE_DOWNLOADS slot cap.
    Idempotent: an already-complete model registers as "done" with no
    thread, and a second caller for an in-flight key joins the existing
    download rather than starting a parallel one.
    """
    key = model_key_for(choice)
    seeded_total = int(choice.size_gb * 1000)
    with _DOWNLOAD_LOCK:
        if is_download_complete(choice):
            _DOWNLOADS[key] = {
                "model_key": key,
                "name": choice.name,
                "progress_pct": 100,
                "downloaded_mb": seeded_total,
                "total_mb": seeded_total,
                "status": "done",
                "error": None,
            }
            return key
        existing = _DOWNLOADS.get(key)
        if existing is not None and existing["status"] in ("queued", "downloading"):
            return key
        _DOWNLOADS[key] = {
            "model_key": key,
            "name": choice.name,
            "progress_pct": 0,
            "downloaded_mb": 0,
            "total_mb": seeded_total,
            "status": "queued",
            "error": None,
        }
        _DOWNLOAD_CHOICES[key] = choice
        _DOWNLOAD_QUEUE.append(key)
        _maybe_start_next()
    return key


def download_status(key: str) -> dict | None:
    """Return a shallow copy of the registry entry for `key`, or None."""
    with _DOWNLOAD_LOCK:
        entry = _DOWNLOADS.get(key)
        return dict(entry) if entry is not None else None


def list_active_downloads() -> list[dict]:
    """Return copies of all in-flight entries: downloading first, then queued.

    Terminal (done/failed) entries are excluded — query those via
    download_status. "queued" entries are ordered by their position in
    the FIFO _DOWNLOAD_QUEUE.
    """
    with _DOWNLOAD_LOCK:
        downloading = [
            dict(e) for e in _DOWNLOADS.values() if e["status"] == "downloading"
        ]
        queued = [
            dict(_DOWNLOADS[k])
            for k in _DOWNLOAD_QUEUE
            if k in _DOWNLOADS and _DOWNLOADS[k]["status"] == "queued"
        ]
        return downloading + queued


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


def _make_progress_tqdm(on_progress: Callable[[str], None]):
    """Build a tqdm subclass that forwards download progress to the UI.

    huggingface_hub drives BOTH of its backends — hf_xet (Rust, parallel
    chunked) and plain HTTP — through `tqdm_class(...).update(nbytes)`,
    so overriding `update` is the one hook that sees every downloaded
    byte regardless of backend. Drawing to stderr stays disabled (a real
    tqdm bar would corrupt the Textual screen); we keep our own byte
    counter (tqdm's own `n` stops counting when `disable=True`) and
    throttle the UI callback to ~2 Hz.
    """
    from huggingface_hub.utils import tqdm as _hf_tqdm

    class _ProgressTqdm(_hf_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True  # never draw to stderr
            super().__init__(*args, **kwargs)
            # `initial` is the resume offset on the HTTP path.
            self._lc_done = int(kwargs.get("initial") or 0)
            self._lc_last_emit = 0.0

        def update(self, n=1):
            self._lc_done += int(n or 0)
            now = time.monotonic()
            if now - self._lc_last_emit >= 0.5:
                self._lc_last_emit = now
                total = int(self.total or 0)
                # Sub-MB files (e.g. config.json) used to render as
                # "0/0 MB (100%)" — integer MB truncation. Show percent
                # alone for those; MB counts only once there are whole MBs
                # to show.
                if total >= 1024 * 1024:
                    pct = min(100, self._lc_done * 100 // total) if total else 0
                    on_progress(
                        f"Downloading: {self._lc_done // (1024 * 1024)}"
                        f"/{total // (1024 * 1024)} MB ({pct}%)"
                    )
                elif total:
                    pct = min(100, self._lc_done * 100 // total)
                    on_progress(f"Downloading: {pct}%")
                else:
                    on_progress("Downloading…")
            return super().update(n)

    return _ProgressTqdm


def _try_hub_download(choice, dest: Path, on_progress: Callable[[str], None] | None) -> bool:
    """Try to download via huggingface_hub's resumable HTTP downloader.

    The hf_xet backend is deliberately disabled (`HF_HUB_DISABLE_XET=1`
    at module import, top of this file): Xet is faster on a clean run
    but cannot resume an interrupted download, which on 10-35 GB GGUFs
    meant quitting LocalCode mid-download lost everything. The HTTP path
    streams into an `.incomplete` sidecar and resumes it via Range
    requests — across retries and across app restarts.

    Returns True on success, raises on hard failure. Returns False if the
    library is not available (caller should fall back).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    if on_progress:
        on_progress(f"Downloading {choice.filename} (HuggingFace, accelerated)...")
    downloaded_path = hf_hub_download(
        repo_id=choice.hf_repo,
        filename=choice.filename,
        # Pin to the entry's revision (commit SHA/tag) when set; defaults to
        # "main" (unchanged behavior). Pinning makes an upstream weight swap
        # under the same filename visible instead of silent.
        revision=getattr(choice, "revision", None) or "main",
        local_dir=str(dest.parent),
        tqdm_class=_make_progress_tqdm(on_progress) if on_progress else None,
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


def detect_llama_cpp_server_command() -> str | None:
    for candidate in ("llama-server", "llama_cpp.server"):
        if shutil.which(candidate):
            return candidate
    return None




def install_python_packages(packages: list[str]) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *packages],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, output or f"Installed {' '.join(packages)}."

