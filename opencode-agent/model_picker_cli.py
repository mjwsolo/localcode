#!/usr/bin/env python3
"""localcode's model picker, for the codex/opencode launchers.

Not a reimplementation: this imports localcode's own catalog modules —
MODEL_GROUPS (display name · maker), recommend() for the ★, and
hf_quants.fetch_quants + fit_badge for level 2 — so the list is identical in
substance to the Textual picker: EVERY catalog model (downloaded or not),
then every quant the HF repo ships, with fit badges and downloaded markers.

Prints the chosen gguf filename on stdout; downloads it first if needed.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from localcode.models_catalog import MODEL_GROUPS, recommend  # noqa: E402
from localcode.hf_quants import fetch_quants, fit_badge  # noqa: E402

MODELS_DIR = Path(os.environ.get("LOCALCODE_MODELS_DIR",
                                 Path.home() / ".local/share/localcode/models"))
RAM_GB = int(os.popen("sysctl -n hw.memsize").read() or 0) // (1 << 30) or 16
GLYPH = {"fits": "✓", "tight": "~", "too_big": "✗"}
BOLD, DIM, RESET, STAR = "\033[1m", "\033[2m", "\033[0m", "\033[33m★\033[0m"

def say(msg: str) -> None:
    print(msg, file=sys.stderr)

def ask(prompt: str) -> str:
    say(prompt)
    sys.stderr.write("> "); sys.stderr.flush()
    return (sys.stdin.readline() or "").strip()

def choose(rows: list[str], title: str) -> int | None:
    while True:
        say("")
        say(f"{BOLD}{title}{RESET}")
        for i, r in enumerate(rows, 1):
            say(f"  {i:2d}) {r}")
        pick = ask(f"{DIM}number · b back · q quit{RESET}")
        if pick == "q": sys.exit(130)
        if pick == "b": return None
        if pick.isdigit() and 1 <= int(pick) <= len(rows):
            return int(pick) - 1

def download(repo: str, filename: str, size_gb: float) -> bool:
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    dest, part = MODELS_DIR / filename, MODELS_DIR / (filename + ".part")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    say(f"  downloading {filename} ({size_gb:.1f} GB) …")
    try:
        with urllib.request.urlopen(url) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or size_gb * 1e9)
            done = 0
            while chunk := r.read(1 << 20):
                f.write(chunk); done += len(chunk)
                sys.stderr.write(f"\r  {done * 100 // total}% of {total / 1e9:.1f} GB   ")
        sys.stderr.write("\n")
        part.rename(dest)
        return True
    except Exception as e:  # noqa: BLE001
        part.unlink(missing_ok=True)
        say(f"  download failed: {e}")
        return False

def main() -> int:
    rec_group = None
    try:
        rec_repo = recommend(RAM_GB).hf_repo
        rec_group = next((g for g in MODEL_GROUPS if g.hf_repo == rec_repo), None)
    except Exception:  # noqa: BLE001
        pass

    while True:
        rows = []
        for g in MODEL_GROUPS:
            star = f" {STAR}" if g is rec_group else ""
            n = len(list(MODELS_DIR.glob("*.gguf"))) and sum(
                1 for q in fetch_quants(g.hf_repo) if (MODELS_DIR / q.filename).exists())
            mark = f" {DIM}· {n} downloaded{RESET}" if n else ""
            rows.append(f"{BOLD}{g.display_name}{RESET} {DIM}· {g.maker}{RESET}{star}{mark}")
        gi = choose(rows, f"Choose a model — {RAM_GB} GB Mac · {STAR} recommended for you")
        if gi is None:
            continue
        g = MODEL_GROUPS[gi]

        quants = fetch_quants(g.hf_repo)
        if not quants:
            say("  could not list quants (offline?) — try another model")
            continue
        qrows = []
        for q in quants:
            here = (MODELS_DIR / q.filename).exists()
            badge = "✓ downloaded" if here else f"{GLYPH[fit_badge(q.size_gb, RAM_GB)]} {fit_badge(q.size_gb, RAM_GB).replace('_', ' ')}"
            qrows.append(f"{q.label:<14} {DIM}·{RESET} {q.size_gb:5.1f} GB {DIM}·{RESET} {badge}")
        qi = choose(qrows, f"{g.display_name} {DIM}· {g.maker} · {g.license}{RESET}")
        if qi is None:
            continue
        q = quants[qi]
        if not (MODELS_DIR / q.filename).exists() and not download(g.hf_repo, q.filename, q.size_gb):
            continue
        print(q.filename[:-5] if q.filename.endswith(".gguf") else q.filename)
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
