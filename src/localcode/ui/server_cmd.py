"""Print the llama-server command localcode itself would launch on THIS machine
for a given GGUF: context size from the RAM tier ladder, KV-cache compression,
flash-attn, batch sizes, checkpoints. Nothing hardcoded in the front ends.

    server_cmd.py <gguf> <port> [alias]   -> one argument per line
    server_cmd.py --ctx <gguf>             -> just the context size
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from localcode.config import load_config
from localcode.runtime import LocalCodeRuntimeGateway


def server_command(gguf: str, port: int, alias: str | None = None, vision: bool = False) -> list[str]:
    """`vision`: load the image projector. The supervisor asks for it only once
    an image is sent in the session (lazy projector), so a text-only session
    never spends its 0.2-3.9 GB."""
    cfg = load_config()
    g = LocalCodeRuntimeGateway(cfg.runtime)
    validate_model_memory(gguf, g._system_ram_gb(), vision=vision)
    cmd = g.llama_server_command(gguf, port)
    _drop_flag(cmd, "--mmproj", 1)
    validate_kv_cache_settings(cmd)
    # Validate AFTER stored tuning and environment overrides have rewritten flags.
    ram_gb = g._system_ram_gb()
    ceiling = min(g._ram_ctx_ceiling(ram_gb), g._kv_aware_ctx_ceiling(gguf, ram_gb), g._model_max_ctx(gguf))
    if g._is_cohere_gguf(gguf):
        ceiling = min(ceiling, g._cohere_ctx_ceiling(ram_gb))
    try:
        requested_ctx = int(cmd[cmd.index("--ctx-size") + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError("Invalid model context setting; choose a smaller context or clear the tuning override.") from exc
    if requested_ctx <= 0:
        raise ValueError("Model context must be a positive token count.")
    # Old presets/tuning should recover automatically on the current machine.
    # Weight-fit validation still refuses models that cannot leave OS headroom.
    cmd[cmd.index("--ctx-size") + 1] = str(min(requested_ctx, ceiling))
    cmd[1:1] = ["--host", "127.0.0.1"]
    # Any web page can reach a localhost port; the key makes the model and its
    # /slots (which can hold prompt text) unusable without the session secret.
    from localcode.ui.auth import server_key
    if server_key():
        cmd += ["--api-key", server_key()]
    if alias:
        cmd += ["--alias", alias]
    # Hidden thinking is a SERVER property, one switch for every front end and
    # every model. Per-wire hacks (chat_template_kwargs, reasoning effort
    # "none", --chat-template-kwargs) silently failed on Muse Glimmer; the
    # server flags worked on every wire. localcode's default is off.
    # Per-model policy from the catalog (models_catalog.ModelChoice.reasoning_control):
    #   "server" / "chat_template" -> the flags below silence it (gemma 4, Qwen 3.x)
    #   "always"                    -> no off switch exists (Muse Glimmer); don't send flags
    #   "none"                      -> no reasoning channel at all
    control = "server"
    try:
        from localcode.models_catalog import by_filename
        choice = by_filename(Path(gguf).name)
        if choice is not None:
            control = choice.reasoning_control or "server"
        # The capability registry has the last word. DiffusionGemma's catalog
        # entry carries the default policy, but the model has no thinking
        # toggle: with --reasoning off it ends generation at canvas position 0,
        # so short answers come back empty and long ones lose their first
        # words (found by the launcher-driven model gate, 2026-10-04).
        from localcode.reasoning_capabilities import ReasoningControl, reasoning_capabilities
        capabilities = reasoning_capabilities(Path(gguf).name)
        if capabilities.control is ReasoningControl.NONE:
            control = "none"
        if capabilities.family == "diffusion":
            # The model always opens a `<|channel>thought` block itself and
            # cannot be told not to (see above). With the parser off that
            # block leaked into every reply as literal `<|channel>thought ...`
            # text; with it on, the thought goes to reasoning_content and the
            # answer after `<channel|>` is the content, like every other model.
            # A reply cut off by max_tokens before `<channel|>` has no content
            # either way: the model had not answered yet.
            cmd += ["--reasoning-format", "auto"]
    except Exception:  # noqa: BLE001, S110
        pass
    mode = os.environ.get("LOCALCODE_INTERNAL_THINKING_MODE") or cfg.runtime.internal_thinking_mode or "off"
    if control in ("server", "chat_template") and str(mode).strip().lower() != "on":
        cmd += ["--reasoning", "off", "--reasoning-budget", "0"]
    # Vision: pass --mmproj only when the projector was asked for (an image was
    # sent) and the sidecar is on disk.
    mm = mmproj_for(gguf) if vision else None
    if mm is not None:
        cmd += ["--mmproj", str(mm)]
    _fit_drafter(g, gguf, cmd)
    # Budget every weight file actually passed to the server, including an
    # explicit user drafter. A 2048-token floor must not override lack of RAM.
    loaded_paths = {Path(gguf)}
    for flag in ("--mmproj", "--model-draft"):
        if flag in cmd:
            loaded_paths.add(Path(cmd[cmd.index(flag) + 1]))
    loaded_weights = sum(p.stat().st_size for p in loaded_paths if p.is_file())
    # Sliding-window caches and the projector's compute buffer do not grow
    # with the context but share the same memory.
    from localcode import gpu_memory
    loaded_weights += g._kv_fixed_bytes(gguf) + (gpu_memory.VISION_COMPUTE_BYTES if "--mmproj" in cmd else 0)
    bpt = g._kv_bytes_per_token(gguf)
    draft = cmd[cmd.index("--model-draft") + 1] if "--model-draft" in cmd else None
    spec = cmd[cmd.index("--spec-type") + 1] if "--spec-type" in cmd else ""
    if draft or spec == "draft-mtp":
        draft_path = draft or gguf
        draft_meta = g._gguf_kv_meta(draft_path)
        if not draft_meta:
            raise ValueError("Cannot estimate draft-model memory from its GGUF header; disable drafting or select a supported drafter.")
        # Gemma assistant borrows target KV; ordinary/DFlash contexts own KV.
        if draft_meta.get("architecture") != "gemma4-assistant":
            draft_bpt = g._kv_bytes_per_token(draft_path, dtypes=("f16", "f16"))
            if spec == "draft-mtp":
                nextn = draft_meta.get("n_mtp_layers", 0)
                if nextn:
                    draft_bpt *= nextn / draft_meta["n_layers"]
                elif not draft:
                    draft_bpt = 0
            bpt = (bpt or 0) + (draft_bpt or 0)
    if bpt:
        total = ram_gb * 1024**3
        reserve = max(3 * 1024**3, int(total * 0.15))
        fit = int((total - loaded_weights - reserve) / bpt) // 2048 * 2048
        if fit < 2048:
            raise ValueError("Insufficient RAM for model weights and minimum context; choose a smaller quantization in /models.")
        # When the weights fit the GPU working set, keep the KV cache inside it
        # too: past it Metal pages or the kernel kills the server mid-turn.
        # Weights already over it (a quant the picker marks tight) keep the
        # RAM-based context.
        ws_free = gpu_memory.working_set_bytes(ram_gb) - loaded_weights - gpu_memory.COMPUTE_RESERVE_BYTES
        if ws_free > 0:
            fit = min(fit, max(2048, int(ws_free / bpt) // 2048 * 2048))
        ctx_index = cmd.index("--ctx-size") + 1
        cmd[ctx_index] = str(min(int(cmd[ctx_index]), fit))
    # Prompt warm-up (ui/warmup.py) reads the first turn's token ids back out
    # of a slot save; the flag only enables the endpoint, nothing is written
    # until the supervisor asks.
    try:
        from localcode.ui import warmup
        if warmup.enabled():
            d = warmup.slot_save_dir()
            d.mkdir(parents=True, exist_ok=True)
            cmd += ["--slot-save-path", str(d)]
    except Exception:  # noqa: BLE001, S110
        pass
    return cmd


def _drop_flag(cmd: list[str], flag: str, nargs: int) -> None:
    while flag in cmd:
        i = cmd.index(flag)
        del cmd[i:i + 1 + nargs]


def _fit_drafter(g, gguf: str, cmd: list[str]) -> None:
    """Leave the catalog's speculative drafter out when it does not fit beside
    the model, a working context and compute buffers in the GPU working set
    (models_catalog.quant_plan uses the same rule for the picker). On a 16 GB
    Mac the 0.4-1.7 GB drafter is the difference between a better quant and a
    worse one; where it fits it stays (1.5-2x decode)."""
    if "--model-draft" not in cmd:
        return
    drafter = g._catalog_drafter_path(gguf)
    draft = Path(cmd[cmd.index("--model-draft") + 1])
    if drafter is None or draft != Path(drafter[1]):
        return  # a user-chosen draft model is the user's call
    from localcode import gpu_memory
    paths = {Path(gguf), draft}
    if "--mmproj" in cmd:
        paths.add(Path(cmd[cmd.index("--mmproj") + 1]))
    weights = sum(p.stat().st_size for p in paths if p.is_file())
    fixed = g._kv_fixed_bytes(gguf) + (gpu_memory.VISION_COMPUTE_BYTES if "--mmproj" in cmd else 0)
    bpt = g._kv_bytes_per_token(gguf) or 0
    if gpu_memory.fit(weights, g._system_ram_gb(), bpt, fixed) != "fits":
        _drop_flag(cmd, "--model-draft", 1)
        _drop_flag(cmd, "--spec-type", 1)
        _drop_flag(cmd, "--spec-draft-n-max", 1)


def catalog_choice(gguf: str):
    """ModelChoice for a gguf: a curated entry, else one minted for a browsed quant."""
    try:
        from localcode.models_catalog import by_filename, choice_for_quant, group_for_filename
        name = Path(gguf).name
        choice = by_filename(name)
        if choice is not None:
            return choice
        g = group_for_filename(name)
        if g is not None:
            return choice_for_quant(g, name, 0.0)
    except Exception:  # noqa: BLE001, S110
        pass
    return None


def mmproj_for(gguf: str) -> Path | None:
    choice = catalog_choice(gguf)
    if choice is None or not getattr(choice, "supports_vision", False):
        return None
    for cand in (getattr(choice, "mmproj_path", None), Path(gguf).parent / (choice.mmproj_filename or "")):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


def context_size(gguf: str) -> int:
    cmd = server_command(gguf, 0)
    return int(cmd[cmd.index("--ctx-size") + 1])


def validate_kv_cache_settings(cmd: list[str]) -> None:
    """Reject unsupported Metal mixed caches before a backend assertion."""
    if "--flash-attn" not in cmd or cmd[cmd.index("--flash-attn") + 1] != "on":
        return
    def value(flag: str) -> str:
        return cmd[cmd.index(flag) + 1].lower() if flag in cmd else "f16"
    key, val = value("--cache-type-k"), value("--cache-type-v")
    mixed = {"q8_0", "turbo2", "turbo3", "turbo4"}
    if key != val and not (key in mixed and val in mixed):
        raise ValueError(
            f"Unsupported mixed KV caches: {key} keys / {val} values. "
            "Use matching cache types, or q8_0 keys with q8_0/TurboQuant values."
        )


def validate_model_memory(gguf: str, ram_gb: float, vision: bool = False) -> None:
    """Reject definite over-capacity loads; passing is not a fit guarantee.

    Include installed projector/drafter weights and OS reserve. Actual KV and
    Metal workspace peaks are measured by the local hardware gate.
    """
    paths = {Path(gguf)}
    projector = mmproj_for(gguf) if vision else None
    if projector:
        paths.add(projector)
    weights = sum(p.stat().st_size for p in paths if p.is_file())
    total = int(ram_gb * 1024 ** 3)
    if total <= 0:
        raise ValueError("Could not determine RAM; run `localcode doctor` before loading a model.")
    reserve = max(3 * 1024 ** 3, int(total * 0.15))
    if weights and weights + reserve >= total:
        raise ValueError(
            f"Insufficient RAM: model weights require {weights / 1024**3:.1f} GiB "
            f"on this {ram_gb:g} GiB Mac before KV cache and Metal workspace. "
            "Choose a smaller model or quantization in /models; close other apps."
        )


if __name__ == "__main__":
    if sys.argv[1] == "--ctx":
        print(context_size(sys.argv[2]))
    else:
        gguf, port = sys.argv[1], int(sys.argv[2])
        alias = sys.argv[3] if len(sys.argv) > 3 else None
        print("\n".join(server_command(gguf, port, alias)))
