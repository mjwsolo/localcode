#!/usr/bin/env python3
"""Render every eval run on disk as one self-contained HTML page.

    python3 evals/dashboard.py [runs_dir ...] [--out evals/runs/dashboard.html] [--open]

Reads <runs_dir>/<run>/results.jsonl (one row per trial, written by run_tasks.py)
and draws: a coverage matrix (which model has which layer of evidence), a per-model
table from each model's latest run, a task-by-model heatmap, tool-call validity,
decode speed where the run recorded server timings, and pass-rate history per
model. No dependencies, no network, nothing leaves the machine.
"""
from __future__ import annotations

import argparse
import html
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Chart palette (validated categorical order, sequential blue, status colours).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"


def load_runs(dirs: list[Path]) -> list[dict]:
    runs = []
    for d in dirs:
        for run in sorted(p for p in d.iterdir() if p.is_dir()):
            f = run / "results.jsonl"
            if not f.is_file():
                continue
            rows = []
            for line in f.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if not rows:
                continue
            name = run.name
            date = name[:13]                       # YYYYMMDD-HHMM
            model = rows[0].get("model") or name.rsplit("-", 1)[-1]
            label = rows[0].get("label") or name[14:].rsplit("-" + model, 1)[0]
            runs.append({"name": name, "date": date, "model": model, "label": label, "rows": rows, "path": run})
    return runs


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def decode_tps(row) -> float | None:
    reqs = row.get("server_requests") or []
    vals = [r["decode_tokens"] / (r["decode_ms"] / 1000) for r in reqs
            if r.get("decode_ms") and r.get("decode_tokens", 0) >= 16]
    return med(vals)


def validity(rows) -> float | None:
    calls = sum(r.get("tool_calls", 0) + r.get("tool_narrated", 0) - r.get("tool_rejected", 0) - r.get("tool_guarded", 0) for r in rows)
    bad = sum(r.get("tool_invalid", 0) + r.get("tool_narrated", 0) for r in rows)
    has_metric = any("tool_validity" in r for r in rows)
    return (calls - bad) / calls if has_metric and calls else None


def summarize(run) -> dict:
    rows = run["rows"]
    return {
        "pass_rate": sum(bool(r.get("pass")) for r in rows) / len(rows),
        "trials": len(rows),
        "tasks": len({r["task"] for r in rows}),
        "validity": validity(rows),
        "wall": med([r.get("wall_s") for r in rows]),
        "steps": med([r.get("steps") for r in rows]),
        "decode": med([decode_tps(r) for r in rows]),
        "reread": med([max(r.get("reread_tokens") or [0]) for r in rows]),
    }


def catalog_models() -> list[tuple[str, str, bool]]:
    """(group key, display name, has vision) from the shipping catalog, if importable."""
    try:
        sys.path.insert(0, str(HERE.parent / "src"))
        from localcode.models_catalog import MODEL_GROUPS  # type: ignore
        return [(g.key, g.display_name, g.supports_vision) for g in MODEL_GROUPS]
    except Exception:
        return []


def gate_receipt(dirs: list[Path]) -> dict:
    """The newest local-gate receipt: scripts/verify_models.py writes it to
    <checkout>/.localcode-gate/receipt.json; a copy kept as <runs>/gate-<sha>.json
    survives the checkout being deleted."""
    candidates = [HERE.parent / ".localcode-gate" / "receipt.json"]
    for d in dirs:
        candidates += sorted(d.glob("gate-*.json"))
    best, best_t = {}, -1.0
    for p in candidates:
        if p.is_file() and p.stat().st_mtime > best_t:
            try:
                best, best_t = json.loads(p.read_text()), p.stat().st_mtime
            except Exception:
                continue
    return best


def _catalog():
    try:
        sys.path.insert(0, str(HERE.parent / "src"))
        import localcode.models_catalog as cat  # type: ignore
        return cat
    except Exception:
        return None


def model_key(alias: str) -> str:
    """Catalog group key for a run's model name (the GGUF stem). Resolved through
    the shipping catalog so the coverage matrix lines up with its rows; the
    string table below is the fallback when the catalog is not importable."""
    cat = _catalog()
    if cat is not None:
        g = cat.group_for_filename(alias if alias.endswith(".gguf") else alias + ".gguf")
        if g:
            return g.key
    a = alias.lower()
    if "diffusiongemma" in a: return "diffusiongemma-26b-a4b"
    if "gemma-4-12b" in a: return "gemma-4-12b"
    if "gemma-4-26b" in a: return "gemma-4-26b-a4b"
    if "qwen3.6-35b" in a: return "qwen-3.6-35b-a3b"
    if "qwen3.8-27b" in a: return "qwen-3.8-27b"
    if "muse" in a: return "muse-glimmer-30b"
    if "north" in a: return "north-mini-code-1.0"
    if "kolibri" in a: return "kolibri-1"
    if "saluki" in a: return "saluki-27b"
    if "openjev" in a: return "openjev"
    return a


def quant_label(model: str) -> str:
    """Q8_K_XL / IQ4_XS / IQ2-mix from a GGUF stem; the catalog's own labeller first."""
    cat = _catalog()
    if cat is not None:
        try:
            lab = cat._quant_label(model if model.endswith(".gguf") else model + ".gguf")
            if lab:
                return lab
        except Exception:
            pass
    if "-UD-" in model:
        return model.split("-UD-")[-1]
    import re
    m = re.search(r"((?:UD-)?I?Q\d[A-Za-z0-9_-]*)$", model)
    return m.group(1) if m else model[-10:]


def gate_by_group(receipt: dict) -> dict[str, dict]:
    """Gate receipts are keyed by ModelChoice key (gemma, qwen38-q8, ...); the
    coverage matrix is keyed by catalog group. Fold every non-turbo4 record of a
    group into one entry whose checks are the union of what its quants passed."""
    cat = _catalog()
    out: dict[str, dict] = {}
    for m in receipt.get("models", []):
        if m.get("turbo4"):
            continue
        gkey = m["key"]
        if cat is not None:
            ch = cat.by_key(m["key"])
            g = cat.group_for_filename(ch.filename) if ch else None
            if g:
                gkey = g.key
        e = out.setdefault(gkey, {"key": gkey, "checks": [], "quants": []})
        for c in m.get("checks", []):
            if c not in e["checks"]:
                e["checks"].append(c)
        e["quants"].append(m["key"])
    return out


def esc(s) -> str:
    return html.escape(str(s))


def fmt(v, kind="") -> str:
    if v is None:
        return '<span class="muted">n/a</span>'
    if kind == "pct":
        return f"{v * 100:.0f}%"
    if kind == "s":
        return f"{v:.0f} s"
    if kind == "tps":
        return f"{v:.1f}"
    if kind == "int":
        return f"{v:.0f}"
    return esc(v)


def status_dot(v: float | None, good=0.9, warn=0.7) -> str:
    if v is None:
        return '<span class="dot none" title="no data"></span>'
    c = GOOD if v >= good else WARN if v >= warn else CRIT
    return f'<span class="dot" style="background:{c}"></span>'


def heatmap(latest: dict[str, dict], tasks: list[str]) -> str:
    models = list(latest)
    cells = []
    for t in tasks:
        row = [f'<th class="rowhead">{esc(t)}</th>']
        for m in models:
            rs = [r for r in latest[m]["rows"] if r["task"] == t]
            if not rs:
                row.append('<td class="cell none" title="not run"></td>')
                continue
            p = sum(bool(r.get("pass")) for r in rs) / len(rs)
            c = GOOD if p == 1 else WARN if p > 0 else CRIT
            tip = f"{esc(m)} / {esc(t)}: {sum(bool(r.get('pass')) for r in rs)}/{len(rs)} pass; " + "; ".join(esc(r.get("notes", ""))[:80] for r in rs)
            row.append(f'<td class="cell" style="background:{c}" title="{tip}"></td>')
        cells.append("<tr>" + "".join(row) + "</tr>")
    head = "<tr><th></th>" + "".join(f'<th class="colhead"><div>{esc(m)}</div></th>' for m in models) + "</tr>"
    return f'<table class="heat">{head}{"".join(cells)}</table>'


def history_svg(runs: list[dict], models: list[str]) -> str:
    W, H, L, B = 760, 230, 44, 28
    by_model = defaultdict(list)
    for r in runs:
        by_model[r["model"]].append((r["date"], summarize(r)["pass_rate"]))
    dates = sorted({r["date"] for r in runs})
    if len(dates) < 2:
        return '<p class="muted">History appears once a model has two or more runs.</p>'
    x = lambda d: L + (dates.index(d) / (len(dates) - 1)) * (W - L - 12)
    y = lambda p: 12 + (1 - p) * (H - B - 12)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="pass rate per run by model">']
    for g in (0, 0.5, 1):
        out.append(f'<line x1="{L}" x2="{W-12}" y1="{y(g):.1f}" y2="{y(g):.1f}" class="grid"/>'
                   f'<text x="{L-6}" y="{y(g)+4:.1f}" class="axis" text-anchor="end">{int(g*100)}%</text>')
    shown = set()
    for d in dates:
        day = d[:8]
        if day in shown:
            continue
        shown.add(day)
        out.append(f'<text x="{x(d):.1f}" y="{H-8}" class="axis" text-anchor="middle">{d[4:6]}-{d[6:8]}</text>')
    for i, m in enumerate(models[:8]):
        pts = sorted(by_model.get(m, []))
        if not pts:
            continue
        col = SERIES[i % 8]
        path = " ".join(f"{'M' if j == 0 else 'L'}{x(d):.1f},{y(p):.1f}" for j, (d, p) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        for d, p in pts:
            out.append(f'<circle cx="{x(d):.1f}" cy="{y(p):.1f}" r="4" fill="{col}" stroke="var(--surface)" stroke-width="2"><title>{esc(m)} {d}: {p*100:.0f}%</title></circle>')
        d, p = pts[-1]
        short = esc(m.split("-UD-")[0][:18])
        anchor = "end" if x(d) > W - 120 else "start"
        dx = -8 if anchor == "end" else 8
        out.append(f'<text x="{x(d)+dx:.1f}" y="{y(p) - 8 + (i % 2) * 20:.1f}" class="label" text-anchor="{anchor}">{short}</text>')
    out.append("</svg>")
    return "".join(out)


def bars(items: list[tuple[str, float | None]], kind: str, vmax: float | None = None) -> str:
    vals = [v for _, v in items if v is not None]
    if not vals:
        return '<p class="muted">No data yet.</p>'
    vmax = vmax or max(vals)
    rows = []
    for name, v in items:
        w = 0 if v is None else (v / vmax) * 100
        col = SERIES[0] if kind != "pct" else (GOOD if v is not None and v >= 0.95 else WARN if v is not None and v >= 0.8 else CRIT)
        rows.append(f'<div class="bar"><span class="barlabel">{esc(name)}</span>'
                    f'<span class="track"><span class="fill" style="width:{w:.1f}%;background:{col}"></span></span>'
                    f'<span class="barval">{fmt(v, kind)}</span></div>')
    return "".join(rows)


def render(runs: list[dict], dirs: list[Path]) -> str:
    receipt = gate_receipt(dirs)
    latest: dict[str, dict] = {}
    for r in sorted(runs, key=lambda r: r["date"]):
        # The newest full run per model; a smoke run of a few tasks never
        # replaces a regression set.
        full = len({x["task"] for x in r["rows"]}) >= 10
        prev = latest.get(r["model"])
        if prev is None or full or len({x["task"] for x in prev["rows"]}) < 10:
            latest[r["model"]] = r
    models = sorted(latest, key=lambda m: -summarize(latest[m])["pass_rate"])
    tasks = sorted({row["task"] for r in runs for row in r["rows"]})
    covered = {model_key(m) for m in latest}
    validity_by_model = {m: summarize(latest[m])["validity"] for m in models}

    # Coverage: for each catalog model, which evidence layer has a result, and
    # what it says. One axis only; metrics live inside the cell, not as columns.
    cat = catalog_models() or [(model_key(m), m, False) for m in models]
    layers = ["No-model checks (CI)", "Gate (loads, acts once)", "Tasks on this machine", "Tasks as a 24 GB Mac", "Tasks as a 16 GB Mac", "Tasks on real 16/24 GB hardware"]
    gate_rows = gate_by_group(receipt) if receipt else {}
    gate_when = (receipt.get("verified_at") or receipt.get("commit", ""))[:10] if receipt else ""
    def cell(kind: str, text: str, tip: str = "") -> str:
        return f'<td class="cov {kind}" title="{esc(tip)}">{text}</td>'
    def task_cell(key: str, label_prefix: str | None) -> str:
        cands = [r for r in runs if model_key(r["model"]) == key and (label_prefix is None and not r["label"].startswith("tier") or label_prefix and r["label"].startswith(label_prefix))]
        full = [r for r in cands if len({x["task"] for x in r["rows"]}) >= 10]
        if not (full or cands):
            return cell("no", "no run")
        r = sorted(full or cands, key=lambda r: r["date"])[-1]
        sm = summarize(r)
        bits = [f"{sm['pass_rate']*100:.0f}% pass"]
        if sm["validity"] is not None: bits.append(f"{sm['validity']*100:.0f}% valid calls")
        if sm["decode"]: bits.append(f"{sm['decode']:.0f} tok/s")
        kind = "yes" if sm["pass_rate"] >= 0.9 else "part"
        return cell(kind, f"{esc(quant_label(r['model']))}<br>" + ", ".join(bits), f"{r['name']}: {sm['trials']} trials over {sm['tasks']} tasks")
    cov_rows = []
    for key, name, vision in cat:
        g = gate_rows.get(key)
        if g:
            checks = g.get("checks", [])
            gtxt = f"pass {esc(gate_when)}<br>" + ", ".join(esc(c) for c in checks)
            gcell = cell("yes" if (not vision or "vision" in checks) else "part", gtxt, "newest gate receipt on disk; quants: " + ", ".join(g.get("quants", [])))
        else:
            gcell = cell("no", "no receipt on disk", "the gate ran on earlier checkouts that were deleted; receipts are now copied into runs/")
        cov_rows.append("<tr><th class=\"rowhead\">" + esc(name) + "</th>"
                        + cell("yes", "every PR", "unit tests, bun tests, Checks against canned servers")
                        + gcell
                        + task_cell(key, None) + task_cell(key, "tier24") + task_cell(key, "tier16")
                        + cell("no", "needs the hardware", "tier runs emulate the picker's choices, not memory pressure")
                        + "</tr>")
    coverage = ('<table class="covtab"><tr><th></th>' + "".join(f"<th>{esc(l)}</th>" for l in layers) + "</tr>" + "".join(cov_rows) + "</table>")

    # Per-model table.
    trs = []
    for m in models:
        s = summarize(latest[m])
        trs.append(f"<tr><td>{status_dot(s['pass_rate'])}{esc(m)}</td><td class='num'>{fmt(s['pass_rate'],'pct')}</td>"
                   f"<td class='num'>{s['trials']} / {s['tasks']}</td><td class='num'>{fmt(s['validity'],'pct')}</td>"
                   f"<td class='num'>{fmt(s['wall'],'s')}</td><td class='num'>{fmt(s['steps'],'int')}</td><td class='num'>{fmt(s['decode'],'tps')}</td>"
                   f"<td class='num'>{fmt(s['reread'],'int')}</td><td class='muted'>{esc(latest[m]['date'])} {esc(latest[m]['label'])}</td></tr>")
    table = ("<table class='models'><tr><th>Model (latest run)</th><th>Pass</th><th>Trials / tasks</th><th>Tool-call validity</th>"
             "<th>Median wall</th><th>Median steps</th><th>Decode tok/s</th><th>Max re-read tokens</th><th>Run</th></tr>" + "".join(trs) + "</table>")

    total_rows = sum(len(r["rows"]) for r in runs)
    overall = sum(bool(x.get("pass")) for r in runs for x in r["rows"]) / total_rows if total_rows else 0
    css = f"""
    :root{{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10)}}
    @media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--surface:#1a1a19;--plane:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10)}}}}
    :root[data-theme="dark"]{{--surface:#1a1a19;--plane:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10)}}
    body{{margin:0;padding:24px 16px;background:var(--plane);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
    main{{max-width:1100px;margin:0 auto}} h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}}
    .sub{{color:var(--ink2);margin:0 0 16px}} .muted{{color:var(--muted)}}
    .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
    .tile{{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:12px}} .tile .v{{font-size:26px;font-weight:600}} .tile .k{{color:var(--ink2);font-size:12px}}
    table{{border-collapse:collapse;width:100%;background:var(--surface);border:1px solid var(--ring);border-radius:8px;overflow:hidden}}
    th,td{{padding:6px 8px;text-align:left;border-bottom:1px solid var(--grid);vertical-align:top}} th{{color:var(--ink2);font-weight:500;font-size:12px}} td.num{{text-align:right;font-variant-numeric:tabular-nums}}
    .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;vertical-align:middle}} .dot.none{{background:var(--axis)}}
    .heat{{width:auto}} .heat .cell{{width:22px;height:22px;padding:0;border:2px solid var(--surface)}} .heat .cell.none{{background:var(--grid)}}
    .heat .colhead div{{writing-mode:vertical-rl;transform:rotate(180deg);font-size:11px;white-space:nowrap;max-height:160px;overflow:hidden}} .rowhead{{font-size:12px;white-space:nowrap}}
    .covtab td.cov{{font-size:12px;text-align:center}} .cov.yes{{background:rgba(12,163,12,.18)}} .cov.no{{background:rgba(208,59,59,.18)}} .cov.part{{background:rgba(250,178,25,.22)}} .cov.na{{color:var(--muted)}}
    .bar{{display:grid;grid-template-columns:220px 1fr 70px;align-items:center;gap:8px;margin:4px 0}} .barlabel{{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .track{{height:10px;background:var(--grid);border-radius:4px;overflow:hidden}} .fill{{display:block;height:100%;border-radius:4px}} .barval{{text-align:right;font-variant-numeric:tabular-nums;font-size:12px}}
    svg .grid{{stroke:var(--grid)}} svg .axis{{fill:var(--muted);font-size:11px}} svg .label{{fill:var(--ink2);font-size:11px}}
    .wrap{{overflow-x:auto}} footer{{margin-top:28px;color:var(--muted);font-size:12px}}
    """
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LocalCode evals</title><style>{css}</style></head><body><main>
<h1>LocalCode evals</h1><p class="sub">{len(runs)} runs, {total_rows} trials, {len(latest)} models, {len(tasks)} tasks. Generated locally from results on disk.</p>
<div class="tiles">
 <div class="tile"><div class="v">{overall*100:.0f}%</div><div class="k">pass rate, all trials</div></div>
 <div class="tile"><div class="v">{len([m for m in models if summarize(latest[m])['pass_rate']==1])}/{len(models)}</div><div class="k">models at 100% on latest run</div></div>
 <div class="tile"><div class="v">{sum(1 for v in validity_by_model.values() if v is not None)}</div><div class="k">models with a tool-call validity number</div></div>
 <div class="tile"><div class="v">{len(cat) - len(covered & {k for k,_,_ in cat})}</div><div class="k">catalog models with no task run</div></div>
</div>
<h2>Coverage: evidence per catalog model</h2><div class="wrap">{coverage}</div>
<h2>Models, latest run each</h2><div class="wrap">{table}</div>
<h2>Tasks by model, latest run</h2><p class="sub">Green all trials passed, yellow some, red none, grey not run. Hover a cell for the grader's note.</p><div class="wrap">{heatmap(latest, tasks)}</div>
<h2>Tool-call validity</h2><p class="sub">Of the calls the model meant to make, the share that executed. Harness refusals excluded.</p>{bars([(m, validity_by_model[m]) for m in models], "pct", 1.0)}
<h2>Decode speed, median tokens per second</h2>{bars([(m, summarize(latest[m])["decode"]) for m in models], "tps")}
<h2>Pass rate history</h2>{history_svg(runs, models)}
<footer>Runs: {", ".join(esc(r["name"]) for r in sorted(runs, key=lambda r: r["date"]))}</footer>
</main></body></html>"""
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", default=[str(HERE / "runs")])
    ap.add_argument("--out", default=str(HERE / "runs" / "dashboard.html"))
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    a = ap.parse_args()
    runs = load_runs([Path(p) for p in a.runs if Path(p).is_dir()])
    if not runs:
        sys.exit("no runs found")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(runs, [Path(p) for p in a.runs if Path(p).is_dir()]))
    print(f"{len(runs)} runs -> {out}")
    if a.open:
        subprocess.run(["open", str(out)], check=False)


if __name__ == "__main__":
    main()
