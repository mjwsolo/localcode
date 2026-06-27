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


def load_overrides(model_filename: str, path: Path | None = None) -> dict[str, str]:
    """Return ``{LOCALCODE_OVERRIDE_*: value}`` for a model, or ``{}``.

    The shape runtime.apply_param_overrides consumes, so a stored
    recommendation behaves exactly like the env-var sweep that produced it.
    """
    entry = _read(path).get(model_filename)
    if not isinstance(entry, dict):
        return {}
    params = entry.get("params", {})
    out: dict[str, str] = {}
    for field, env_key in _FIELD_TO_ENV.items():
        if params.get(field) is not None:
            out[env_key] = str(params[field])
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
    provenance (machine, pass-rate, tok/s) for auditability.
    """
    p = path or store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _read(p)
    clean = {k: v for k, v in params.items() if k in _FIELD_TO_ENV and v is not None}
    data[model_filename] = {"params": clean, "meta": meta or {}}
    p.write_text(json.dumps(data, indent=2, sort_keys=True))
    return p
