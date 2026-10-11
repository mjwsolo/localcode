"""Per-machine model launch recommendations — the output of the model-opt loop.

The offline optimizer (dev/eval/model_opt.py) sweeps quants/params, scores each
combo on the eval suite, and writes the winner here. At launch,
``runtime.llama_server_command`` consults ``load_overrides`` so the tuned
params are applied automatically.

Default-safe: if the store doesn't exist (nobody has run the optimizer with
``--apply``), ``load_overrides`` returns ``{}`` and launch behaviour is
unchanged. Explicit ``LOCALCODE_OVERRIDE_*`` env vars always win over a
stored recommendation (see ``runtime.apply_param_overrides``).

The store is keyed by model GGUF filename. Each entry records the winning
launch params plus provenance (the machine + score it was tuned on) so a
recommendation tuned on a different Mac is at least auditable.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# Maps a recommendation field to the env var runtime understands.
_FIELD_TO_ENV = {
    "n_gpu_layers": "LOCALCODE_OVERRIDE_NGL",
    "n_ctx": "LOCALCODE_OVERRIDE_NCTX",
    "n_threads": "LOCALCODE_OVERRIDE_THREADS",
    "n_batch": "LOCALCODE_OVERRIDE_BATCH",
}


def store_path() -> Path:
    """Location of the recommendations store (overridable for tests)."""
    override = os.environ.get("LOCALCODE_RECOMMENDATIONS_PATH")
    if override:
        return Path(override)
    base = Path(
        os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    ) / "localcode"
    return base / "model_recommendations.json"


def _read(path: Path | None = None) -> dict[str, Any]:
    p = path or store_path()
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, ValueError, OSError):
        return {}


def hardware_identity() -> dict[str, str]:
    """Recommendations apply only to the hardware on which they were measured."""
    identity = {"machine": platform.machine(), "system": platform.system()}
    for name in ("hw.memsize", "machdep.cpu.brand_string"):
        try:
            identity[name] = subprocess.check_output(
                ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL, timeout=2,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return {}  # do not treat two unknown machines as a match
    return identity


def load_overrides(model_filename: str, path: Path | None = None) -> dict[str, str]:
    """Use validated, same-machine recommendations; explicit env still wins."""
    data = _read(path)
    entry = data.get(model_filename) if isinstance(data, dict) else None
    if not isinstance(entry, dict):
        return {}
    identity = hardware_identity()
    if not identity or entry.get("hardware") != identity:
        return {}
    params = entry.get("params")
    if not isinstance(params, dict):
        return {}
    ranges = {"n_gpu_layers": (0, 999), "n_ctx": (2048, 1048576),
              "n_threads": (1, 256), "n_batch": (1, 8192)}
    out = {}
    for field, env_key in _FIELD_TO_ENV.items():
        value = params.get(field)
        if value is None:
            continue
        if type(value) is not int or not ranges[field][0] <= value <= ranges[field][1]:
            return {}  # do not partially apply an invalid profile
        out[env_key] = str(value)
    return out


def save_recommendation(
    model_filename: str,
    params: dict[str, Any],
    meta: dict[str, Any] | None = None,
    path: Path | None = None,
) -> Path:
    """Write/replace the recommendation for one model. Returns the store path.

    ``params`` keys are the canonical fields (n_gpu_layers / n_ctx /
    n_threads / n_batch); only the ones present are stored. ``meta`` carries
    provenance (machine, pass-rate, tok/s) for auditability. Hardware identity
    is captured automatically; it cannot be overridden through metadata.
    """
    p = path or store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _read(p)
    if not isinstance(data, dict):
        data = {}
    clean = {k: v for k, v in params.items() if k in _FIELD_TO_ENV and v is not None}
    data[model_filename] = {"params": clean, "meta": meta or {}, "hardware": hardware_identity()}
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", dir=p.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p
