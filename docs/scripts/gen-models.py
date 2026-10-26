#!/usr/bin/env python3
"""Generate docs/src/data/models.json from localcode's own catalog.

Everything in the memory chooser comes from localcode itself: the fit rule
is localcode.fit (files plus measured run-time memory inside the GPU working
set, the rule the picker and recommend() use), the recommended pick is
recommend(ram), and every size/active-param string is the catalog's own.

Run with the interpreter that has localcode installed:
  ~/.local/share/uv/tools/localcode/bin/python docs/scripts/gen-models.py
"""
import json, os, pathlib, re
# Size every tier by the documented GPU working-set fractions, not by the Mac
# that happens to run this script.
os.environ["LOCALCODE_RAM_GB"] = "128"
from localcode import models_catalog as m
from localcode import fit as F

NO_AUTO = m._NO_AUTO_RECOMMEND_ARCHS
TIERS = [16, 24, 32, 36, 48, 64, 96, 128]  # real Apple Silicon unified-memory options (>=16 GB floor)

def group_of(choice):
    for g in m.MODEL_GROUPS:
        if g.hf_repo == choice.hf_repo:
            return g
    return None


# Which vendor mark to show. Keyed on the catalogue's own model family so a
# new Gemma/Qwen variant picks up the right logo without another edit here.
_LOGOS = {"gemma4": "gemma", "qwen": "qwen", "cohere": "cohere", "meta": "meta"}


def logo_for(group):
    if group is None:
        return None
    fam = (group.family or "").lower()
    key = group.key.lower()
    for needle, logo in _LOGOS.items():
        if needle in fam or needle in key:
            return logo
    if "north" in key:
        return "cohere"
    if "muse" in key or "glimmer" in key:
        return "meta"
    return None

def facets(choice):
    """Split the catalogue's strings into labelled buckets for the web table.

    Everything is derived from `choice.name` and `choice.active_params`; no
    numbers are introduced here that the catalogue does not already state.
    """
    ap = choice.active_params or ""
    name = choice.name

    quant = None
    mq = re.search(r"\(([^)]+)\)\s*$", name)
    if mq:
        quant = mq.group(1)
    base = re.sub(r"\s*\([^)]*\)\s*$", "", name)

    # Total parameters: the first NNB in the model name (35B-A3B -> 35B).
    mt = re.search(r"(\d+(?:\.\d+)?)B", base)
    total = f"{mt.group(1)}B" if mt else None

    # Active parameters: the first NNB in the catalogue's active_params string.
    ma = re.search(r"(\d+(?:\.\d+)?)B", ap)
    active = f"{ma.group(1)}B" if ma else None

    low = ap.lower()
    if "diffusion" in low:
        kind = "Diffusion MoE"
    elif "dense" in low:
        kind = "Dense"
    else:
        kind = "MoE"

    # The parenthetical detail the catalogue gives, e.g. "top-8 of 128 experts".
    md = re.search(r"\(([^)]+)\)", ap)
    detail = md.group(1) if md else None
    if detail and detail.lower() == "dense":
        detail = None

    return {"base": base, "quant": quant, "total": total,
            "active": active, "kind": kind, "detail": detail}


# Reference-only rows: bigger quants that exist upstream and fit the largest
# Macs, listed so the high-RAM tabs show what those machines can actually run.
# These are NOT in the product catalogue (they are multi-file GGUF splits the
# app does not download yet), so they carry no `recommended` weight — they are
# here to inform, sized from the real upstream files.
_qwen = m.by_key("qwen38")
_qwen_maker = group_of(_qwen).maker if group_of(_qwen) else "—"
_qwen_logo = logo_for(group_of(_qwen))
REFERENCE_MODELS = [
    {
        "name": "Qwen 3.8 27B (BF16, full)",
        "base": "Qwen 3.8 27B",
        "quant": "BF16, full",
        "total": "27B",
        "active": "27B dense (hybrid attention + Mamba-2 SSM, 1 MTP layer)",
        "kind": "Dense",
        "detail": "full-precision weights",
        "file": "Qwen3.8-27B-BF16-00001-of-00002.gguf",
        "size": 54.7,
        "maker": _qwen_maker,
        "logo": _qwen_logo,
        "experimental": False,
    },
]

out = []
for ram in TIERS:
    budget = round(F.gpu_budget_gb(ram), 1)
    rec = m.recommend(ram)
    fits = sorted((c for c in m.CHOICES if F.fit(c.size_gb, ram, c.filename) == "fits"), key=lambda c: c.size_gb)
    models = [{
        "name": c.name,
        **facets(c),
        "file": c.filename,
        "size": c.size_gb,
        "active": c.active_params,
        "maker": (group_of(c).maker if group_of(c) else "—"),
        "logo": logo_for(group_of(c)),
        "experimental": c.architecture in NO_AUTO,
    } for c in fits]
    # Fold in any reference-only model that fits this tier's budget.
    models += [dict(r) for r in REFERENCE_MODELS if F.fit(r["size"], ram, r["file"]) == "fits"]
    models.sort(key=lambda mm: mm["size"])
    out.append({
        "ram": ram,
        "label": f"{ram} GB" + ("+" if ram == TIERS[-1] else ""),
        "budget": budget,
        "recommended": rec.filename,
        "models": models,
    })

p = pathlib.Path(__file__).resolve().parents[1] / "src/data/models.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=2) + "\n")
print(f"wrote {p} — {len(out)} tiers, {len(m.CHOICES)} catalog entries")
