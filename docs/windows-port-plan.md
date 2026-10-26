# Windows port plan

Status today: Apple silicon only. `setup.py:61` forces the wheel tag `macosx_13_0_arm64`, `entrypoint.py:31-40` and `diagnostics.py:21-34` refuse everything else, and both bundled binaries are Mach-O.

## 1. Audit

### Binaries and builds
- llama-server: built from `llama-cpp-turboquant/` with Metal only (`.github/workflows/turboquant.yml:31-52`, `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`, runner macos-latest). Needs new jobs for CUDA, Vulkan and CPU on windows-latest (MSVC).
- Patches: 0008 and 0010 touch only `ggml-metal` and are inert on Windows (they must still apply cleanly). 0001 (TurboQuant KV) already carries CUDA kernels (about 100 `ggml-cuda` diff entries) and 20 `ggml-vulkan` references, so turbo KV can build on both, but it is unverified. 0005 adds `ggml-cuda/diffusion-sampling.cu*`. 0002, 0003, 0006, 0007, 0009 are backend neutral. 0004 is bench scripts only. Risk: turbo2/3/4 CUDA and Vulkan paths are untested.
- UI: `scripts/build_ui_binary.sh` is bash (`/tmp/localcode-ui-build` default, `set -euo pipefail`, `git apply`, python3 heredoc). It builds `packages/cli` with `bun run script/build.ts --single`. Windows needs the fork's build script to target `bun-windows-x64` (and arm64), output `localcode-ui.exe`, plus a Windows-runnable wrapper (run the same script under Git Bash on windows-latest first).
- Wheel: `setup.py:44-55` copies only `bin/llama-server` and `bin/localcode-ui` and lists `bin/*.dylib`; `MANIFEST.in:1-2` names the same files. `ci.yml:137-144` asserts the macOS wheel tag. Plan: per-platform wheels (`win_amd64`, `win_arm64`), a `LOCALCODE_TARGET` env in `setup.py` choosing the tag and binary names (`.exe`, plus CUDA runtime DLLs `ggml-cuda.dll`, `cudart64_*`, `cublas64_*`). CUDA DLLs are several hundred MB, so ship CPU+Vulkan in the main wheel and fetch the CUDA backend as a second wheel (`localcode-cuda`) or an opt-in download (must follow the no-silent-downloads rule: show size, confirm). `bootstrap.py:217-223` (`download_turboquant_binary`) only knows macos-arm64 and linux-x86_64 and pins a stale 0.1.8.

### Platform calls (Python)
- Process lifecycle: `server_manager.py:483` `start_new_session=True`, `:642,736,740,832` `os.killpg`/`os.getpgid`, `:805,935` `ps -p ... -o command=`, `:925` `lsof -ti :port`, `:207-232` `vm_stat`. `ui/diagnostics.py:32` and `diagnostics.py:65` use `ps -axo`. Windows replacements: `CREATE_NEW_PROCESS_GROUP`, `taskkill /T /F` or a Job Object with kill-on-close, `psutil` for process lookup and port owner (add `psutil` dependency, also gives free memory).
- Process guard: `ui/process_guard.py:24,50` use `pass_fds` and `start_new_session`; `signal.SIGHUP` (`:35`, `launch.py:462`, `process_guard.py`) does not exist on Windows. Lease fds do not pass via `pass_fds`. Use a Job Object (kill on close) to give the same "child dies with supervisor" guarantee.
- Single-instance lock: `ui/supervisor.py:1298-1303` and `transfer.py:23,38` use `fcntl.flock`. Replace with `msvcrt.locking` or a named mutex behind one small helper; handle inheritance explicitly (the model inherits the lease today).
- `ui/launch.py:441`, `ui/auth.py:55`, `paths.py:76-154` rely on `chmod 0o600/0o700`; on Windows these are no-ops, so per-user secrecy (`auth.json` API keys) needs `%LOCALAPPDATA%` ACL defaults or `icacls`. `paths.py:82` is already best effort.
- Symlinks (`supervisor.py:240,755,807`, `ui/launch.py:60`) need Developer Mode on Windows; fall back to copy.
- Paths: `~/.local/share/localcode` hardcoded at `bootstrap.py:185,229`, `config.py:138`, `model_files.py:21`, `models_catalog.py:20`, `ui/supervisor.py:1291,1299`, `ui/launch.py:132-133,217`, `recommendations.py:42`; `/tmp/localcode-lookup.bin` at `runtime.py:539`; `/private` handling at `paths.py:267`, `launch.py:136`. Use `platformdirs`-style resolution in one place (`paths.py`) and `tempfile.gettempdir()`.
- macOS only tools: `osascript` (`performance.py:73-177` with LaunchDaemon and `iogpu.wired_limit_mb`, `ui/supervisor.py:1101`), `say` (`supervisor.py:1098,1105`, `voice.py:780`), `afplay` sounds (`config.py:164`), `ps comm=` terminal bundle walk (`voice.py:366-400`), `system_profiler` (`performance.py:405`), `launchctl`, `footprint`, `memorystatus_control` via ctypes (`memory_guard.py:26-200`), `kern.memorystatus_vm_pressure_level` (`memory_guard.py:54`), `thermal.py:64`. The wired-cap and pressure-kill logic is Metal specific; gate it behind a backend check.
- Shell: the agent bash tool is the UI's own; `plugin/localcode.ts:192-209` tells the model it has a persistent bash shell, `:250` `TEMP_RE` assumes POSIX temp paths, `:133` `execFileSync` of argv is fine. On Windows the UI bash tool needs Git Bash (or PowerShell tool description). `bash-guard.test.ts` and `bashDescription` need Windows variants.

### Memory and hardware detection
- RAM: `sysctl hw.memsize` in at least six places (`performance.py:31,116,397`, `runtime.py:1060`, `models_catalog.py:657`, `memory_guard.py:198,258`, `voice.py:339`, `events.py:297`, `recommendations.py:61`); Linux fallback reads `/proc/meminfo`; unknown falls back to 16 GB. Consolidate into one `hardware.py` (psutil, or GlobalMemoryStatusEx via ctypes).
- GPU: `performance.py:_detect_gpu_summary` already tries `nvidia-smi` first, then `system_profiler`. Tiering (`performance.py:372-390`) and `runtime.py:_system_ram_gb` use unified memory as the budget. On Windows the budget must be VRAM for CUDA/Vulkan dGPU (`nvidia-smi --query-gpu=memory.total,memory.free`, or DXGI `DedicatedVideoMemory` via ctypes for AMD/Intel), with expert offload (`-ot exps=CPU`, already used at `runtime.py:498-501`) to spill MoE experts to system RAM. iGPU: budget is shared system RAM, capped near half. CPU-only: use RAM minus headroom and `-ngl 0`.
- Fit and context: `runtime.py:576-600` (`-b/-ub`, threads `max(4, min(cpu,12))` on Darwin, 2..8 else), `runtime.py:673` `-fit off`, `models_catalog.py` picker budget and the 0.5.39 working-set fit assume unified memory. Add a `Backend` object (metal, cuda, vulkan, cpu) giving `weights_budget`, `kv_budget`, `needs_wired_guard`. Without it, 24 GB VRAM plus 64 GB RAM reads as a 64 GB machine.

### Catalog and llama.cpp flags
- MTP drafters (Gemma 4) and DFlash drafters (Qwen 3.6, Muse) in `models_catalog.py:66-97` are plain GGUFs; the flags `--spec-type draft-mtp|draft-dflash` (`runtime.py:543-566`) are backend independent, but speedups were measured on Metal only. Verify on CUDA before enabling; fall back to no drafter on Vulkan if it errors.
- Turbo KV types (`config.py:134-135`, `runtime.py:505-517`) need patch 0001's CUDA/Vulkan kernels; default q8_0/q8_0 elsewhere is safe. Mixed K/V flash attention (patch 0008 is Metal) may be missing on CUDA/Vulkan: guard with a per-backend allowed list.
- `--flash-attn on` (`runtime.py:469`), `--load-mode mmap` (`:493`), `-ngl 999`: on CUDA prefer `--load-mode` default with full offload; mmap of a multi-GB file works on Windows but pages in slowly from disk. Batch size (`runtime.py:587`) is `2048 if Darwin else 128`; CUDA wants 2048.

### UI
- `plugin/modes.ts:318` copies with `pbcopy`; use `clip.exe`/PowerShell `Set-Clipboard` or OSC 52. Voice: `voice.py:285,335` return "macOS-only"; `pywhispercpp`, `sounddevice`, `numpy`, `piper-tts` are gated to `sys_platform=='darwin'` in `pyproject.toml:42-45`. PortAudio via sounddevice and piper work on Windows; whisper.cpp wheel needs a CPU/CUDA build. TTS fallback: SAPI instead of `say`. Mic permission walk (`voice.py:366`) is macOS only. ffmpeg `avfoundation` (`supervisor.py:942`) becomes `dshow`; venv `bin/pip` (`:925`) becomes `Scripts\pip.exe`. Keybindings, colors and image paste come from the opencode fork; test in Windows Terminal and conhost (no truecolor or Kitty keyboard there), Alt modifiers and Ctrl+V.

### Install paths, entry point
- `pyproject.toml:13-14` classifiers (add Windows), `localcode`/`lc` scripts work on Windows (exe shims). Config at `~/.localcode` (`paths.py:14`) is fine; data dir should become `%LOCALAPPDATA%\localcode`.

### Tests assuming macOS
Darwin, sysctl, chmod, tmp and symlink assumptions in `tests/test_config.py`, `test_thermal.py`, `test_compatibility.py`, `test_comprehensive_binary.py`, `test_ui_launch.py`, `test_runtime.py`, `test_comprehensive_project_trust.py`, `test_comprehensive_machines.py`, `test_resumable_transfer.py`; `tests/ui/*.test.ts` bun tests (`ci.yml:86` macos-15 only); `ci.yml` jobs for Intel rejection and macOS install matrix. Process-spawning tests (`test_ui_process_guard.py`, `test_ui_shutdown.py`, `test_ui_server_lifecycle.py`) need Windows variants.

## 2. Phases

### Phase 1: Python on Windows CI (about 1 week)
- Add `comprehensive-windows` (windows-latest, py3.10/3.13) to `ci.yml`. Mark macOS-only tests with a `requires_darwin` marker in `conftest.py`.
- Remove import-time POSIX imports (`fcntl`, `signal.SIGHUP`) behind a small `compat.py` (locks, process groups, kill tree, signals).
- Replace `ps`/`lsof`/`vm_stat`/`sysctl` with psutil in one `hardware.py` and `procs.py`.
- Smoke test: build a CPU-only `llama-server.exe` once, cache it, run a tiny GGUF (e.g. a 15M stories model) through `LOCALCODE_SERVER_FLAVOR=vanilla` path to prove spawn, health, completion, teardown, lease cleanup. The vanilla flavor branch at `runtime.py:444` already omits fork flags.

### Phase 2: CUDA and Vulkan builds (about 1.5 weeks)
- New workflow `windows-server.yml`: matrix cpu, vulkan (Vulkan SDK via `humbletim/setup-vulkan-sdk`), cuda (`Jimver/cuda-toolkit`, `-DCMAKE_CUDA_ARCHITECTURES="75;86;89;120"`). Apply all 10 patches; fail the job if any hunk rejects. CPU/Vulkan static; CUDA ships DLLs.
- Compile only on hosted runners; CUDA cannot run there, so only build and `--version`. Publish `llama-server-windows-*` artifacts.

### Phase 3: UI binary (about 1 week)
- Extend `build_ui_binary.sh` with `--target bun-windows-x64|arm64` and run on windows-latest under Git Bash; verify `localcode-ui.exe` starts, renders in Windows Terminal, talks to a stub server.
- Swap `pbcopy`, bash tool description, `TEMP_RE`, and spawn code in the plugin; bun tests on windows-latest.
- Wheel: `setup.py` per-platform tags and binary names; `ci.yml` build job asserts the Windows wheel contents; add install job on windows-latest (pip and uv).

### Phase 4: Hardware detection and fit (about 1.5 weeks)
- `hardware.py`: RAM, VRAM (nvidia-smi, DXGI), iGPU/dGPU, CPU cores; unit tests with fixtures for NVIDIA 8/12/24 GB, AMD Vulkan, iGPU, CPU-only.
- `Backend` selection: prefer CUDA when an NVIDIA GPU and driver are found, else Vulkan, else CPU; clear message and manual override in the UI picker (not an env var).
- Rework fit, context ceiling and slot count for discrete VRAM plus RAM offload; disable Metal-specific guards (wired cap, pressure monitor); replace with free-VRAM checks before load and a retry with fewer layers on OOM.
- Per-backend flag allow-list (KV types, drafters, batch size, flash attention).

### Phase 5: Real-GPU verification (about 1 week plus iteration)
- Windows terminal UX pass, eval run per backend, SmartScreen behaviour for unsigned exes.
- Rent a Windows GPU VM (or use a PC): RTX 30/40 class, plus an AMD or Intel Vulkan box if possible.
- Live runs per the model-support rule: load, generate, tools, vision, regression on the shipped binary for every catalog model that the recommender can choose. Measure MTP/DFlash and turbo KV on CUDA before enabling.

## 3. Risks
- CUDA wheel size exceeds the PyPI default limit.
- Vendored tree plus patches may fail on MSVC/nvcc; `upstream-bump.yml` then needs Windows build gates.
- Unsigned binaries flagged by SmartScreen.
- Fork bash tool and ConPTY behaviour may need fork work.
- Turbo KV and drafter gains unverified off Metal.
- No GPU CI runners; GPU validation stays manual.

## 4. Effort
About 5 to 7 engineer-weeks end to end: phases 1 to 3 roughly 3.5 weeks, phase 4 about 1.5, phase 5 about 1 plus fix time. Ship as alpha pre-releases, CPU and Vulkan first.
