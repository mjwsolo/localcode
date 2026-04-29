"""Single source of truth for UI colors.

Goal: stop scattering hex literals (`#5f87ff`, `#6ab76a`, `#c88afa`, …)
across screens, widgets, and renderers. Every UI color reference should
go through this module so:

  * Designers (or just future-us) can change the look in ONE place.
  * Light/dark/high-contrast themes become a config flip, not a rewrite.
  * Code reviews can spot "is this color from the palette?" instantly.

Inspired by:
  * agent (TypeScript/Ink) — `src/utils/theme.ts` (semantic keys, theme
    swap, ~15 core names)
  * OpenAI agent TUI (Rust/ratatui) — capability-graded palette per
    terminal class
  * terminal coding tools (Go) — JSON-defined themes with text-strong/weak/weaker
    layered variants

Our take: a flat dict of ~15 semantic names mapped to hex strings.
Three accessors:

    from .theme import C, on, dim
    "[bold {}]Setup[/]".format(C.primary)        # → "#5f87ff"
    line.append("✓", style=C.success)
    panel.styles.background = on.surface
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Palette ──────────────────────────────────────────────────────────
# Hex names are kept short for ergonomic use in f-strings and rich markup
# tags. The full semantic name lives on the dataclass field; downstream
# code never inlines hex.
@dataclass(frozen=True)
class Palette:
    # Brand — ONLY thing that carries colour for non-semantic UI.
    # Anything else uses bold/dim/glyphs to convey meaning.
    primary:        str = "#5f87ff"   # the LocalCode blue — brand, ❯ prompt, tool markers
    # Semantic — used only when the meaning IS "good/bad/warn"; never
    # for decoration. Code that wants a "highlight" should use bold,
    # not a colour.
    success:        str = "#6ab76a"   # ✓ download done, +diff additions
    warning:        str = "#e69947"   # ⚠ cautionary — toned to match the
                                      # visual weight of success/error;
                                      # `#ffb86c` was too bright at L=71
    error:          str = "#ff5874"   # ✗ errors, -diff removals
    # Removed 2026-04-25 (palette consolidation):
    #   accent       (#c88afa, violet) — skills, slash-menu selection
    #   accent_pink  (#f06292, pink)   — slash-menu desc, markdown code
    #   info         (#87afff, cool blue) — never had a real caller
    # All replaced with `bold`/`dim` modifiers so weight contrast
    # carries the highlight without competing with the brand colour.
    # Text levels (agent / terminal coding tools pattern: 4 weight tiers)
    text:           str = "#e8e8e8"   # main foreground prose
    text_muted:     str = "#808080"   # secondary / dim labels
    text_subtle:    str = "#5a5a5a"   # tertiary, easy-to-skip detail
    text_inverse:   str = "#1c1c1c"   # foreground when ON a light bg
    # Surfaces — these aren't actually used as backgrounds anymore.
    # The TUI uses `ansi_default` (terminal-palette) via the
    # `textual-ansi` theme set in tui/app.py — see the explanation
    # in tui/styles/app.tcss. These hex strings are kept only for
    # the rare consumer that imports `C.background` / `C.surface`
    # for a non-TUI render (logs / docs / promptfoo screenshots).
    # If any user-visible UI is reading these for an actual
    # rendered background, that's a bug — route through ANSI instead.
    background:     str = "#1e1e1e"   # legacy, non-rendering
    surface:        str = "#1e1e1e"   # legacy, non-rendering
    border:         str = "#5f87ff"   # picker boxes — same as primary so
                                      # selection state matches surrounding chrome
    # Diff
    diff_add:       str = "#6ab76a"   # green — same family as success
    diff_remove:    str = "#ff5874"   # red — same family as error


# Default (dark) palette — change THEME to swap.
THEME: Palette = Palette()


# ── Accessors ────────────────────────────────────────────────────────
# `C` is shorthand for "color" — the most common consumer pattern is
# inline use in f-strings, where `{C.primary}` reads naturally.
C: Palette = THEME

# `on` namespaces "colors used as a background", though for a flat
# palette it's currently identical to `C`. Kept separate so callers
# can switch to per-surface variants later (e.g. on.raised_surface).
on: Palette = THEME


def dim(color: str) -> str:
    """Return the rich style string `dim {color}` — used everywhere we
    want a softer version without inventing another palette entry."""
    return f"dim {color}"


def bold(color: str) -> str:
    """Return the rich style string `bold {color}`."""
    return f"bold {color}"


# ── Migration helpers ────────────────────────────────────────────────
# When porting a hex literal that appears in many files, prefer the
# semantic name. If you find a hex in our codebase that ISN'T mapped
# here, decide whether it deserves a new palette entry or whether it
# should map to an existing one. Avoid inventing one-off colors.

_LEGACY_HEX_TO_SEMANTIC = {
    "#5f87ff":  "primary",
    "#6ab76a":  "success",
    "#c88afa":  "accent",
    "#f06292":  "accent_pink",
    "#ff5874":  "error",
    "#87afff":  "info",
    "#ffb86c":  "warning",
}


def lookup_legacy(hex_value: str) -> str | None:
    """For migration scripts: given a hex literal, return the semantic
    palette key (or None if the color isn't in the palette)."""
    return _LEGACY_HEX_TO_SEMANTIC.get(hex_value.lower())


# ── Theme switching scaffolding ──────────────────────────────────────
# Light theme stub — fill in when we ship light mode. The shape mirrors
# Palette so the swap is a single assignment.
LIGHT: Palette = Palette(
    primary="#1f5fff",
    success="#16a34a",
    warning="#d97706",
    error="#dc2626",
    text="#1c1c1c",
    text_muted="#525252",
    text_subtle="#a3a3a3",
    text_inverse="#fafafa",
    background="white",
    surface="#f5f5f5",
    border="#1f5fff",
    diff_add="#16a34a",
    diff_remove="#dc2626",
)


def set_theme(name: str) -> None:
    """Swap the active theme. Currently 'dark' (default) or 'light'.
    Mutates the module-level `C` and `on` so existing imports see
    the new colors after the call returns."""
    global C, on, THEME
    if name == "light":
        THEME = LIGHT
    else:
        THEME = Palette()  # dark default
    C = THEME
    on = THEME
