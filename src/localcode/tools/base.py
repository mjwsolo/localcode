"""Shared types for tool modules.

Each tool lives in its own file under src/localcode/tools/ and exports two
symbols:

  SCHEMA:   dict — the OpenAI-compatible function schema handed to the
            model. See TOOL_SCHEMAS usage in src/localcode/agent.py.

  execute:  callable (ctx: ToolContext, args: dict) -> str — invoked
            when the model picks this tool. Returns the result string
            that's fed back as the `tool` message.

Per-tool files + co-located schema+exec is the minimal-agent / terminal coding tools Agent
SDK convention. Makes adding a tool a one-file change, keeps description
next to the code that implements it, and enables per-tool unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..app import LocalCodeApp
    from ..output import OutputManager


@dataclass
class ToolContext:
    """Everything a tool might need to do its job.

    Keep this deliberately small — if a tool can be done with just
    `repo` (Path) and `args` (dict), prefer that. Reach for `app` or
    `out` only when you need the full agent state or the TUI output
    manager.
    """
    app: "LocalCodeApp"
    out: "OutputManager"

    @property
    def repo(self) -> Path:
        return self.app.repo_root

    def resolve_path(self, raw: str) -> Path:
        """Resolve a tool `path` argument, healing corrupted repo prefixes.

        Small quantized models corrupt long absolute paths when they copy
        them between calls. Observed live (2026-07-13): the model turned the
        workdir `…/localcode-evals/20260713-024502/tempconv` into
        `…/localcode-evals/2026-0713-0245-02/tempconv` — hyphens
        hallucinated into the timestamp — and every subsequent write landed
        in a phantom tree while the real project stayed empty. Same class of
        fix as fuzzy old_string matching in edit_file: if an absolute path
        is NOT under repo_root but its leading components are a near-match
        of repo_root's (difflib ratio ≥ 0.8 per component), remap that
        prefix onto the real repo_root. Anything else passes through
        untouched — reads intentionally have full filesystem access.

        NOTE: this method does NOT contain the path. Write tools must call
        `resolve_write_path()` instead, which applies the same healing and
        then refuses anything that lands outside the project root.
        """
        p = self.repo / raw
        try:
            root = self.repo.resolve()
            rp = p.resolve()
            if root == rp or root in rp.parents:
                return p  # already inside the project — nothing to heal
            root_parts = root.parts
            path_parts = rp.parts
            if len(path_parts) <= len(root_parts):
                return p
            import difflib
            for a, b in zip(root_parts, path_parts):
                if a == b:
                    continue
                if difflib.SequenceMatcher(None, a, b).ratio() < 0.8:
                    return p  # genuinely different location — respect it
            healed = root.joinpath(*path_parts[len(root_parts):])
            try:
                from ..events import emit as _emit
                _emit("path_remap", raw=str(raw)[:200], healed=str(healed)[:200])
            except Exception:
                pass
            return healed
        except Exception:
            return p

    def resolve_write_path(self, raw: str) -> Path:
        """Resolve a WRITE target: heal first, then enforce containment.

        Healing runs exactly as it does for reads (small quants mangle long
        absolute paths, and a mangled path that near-matches repo_root is
        remapped onto it). Containment is applied to the healed result, so
        the heal feature keeps working while `/Users/victim/.zshrc`,
        `../../../etc/hosts`, and a repo-internal symlink pointing outside
        all raise `PathContainmentError`.

        Escape hatches, in order:
          * the agent's notebook scratch dir (sanctioned, may live outside
            the project root), and
          * paths the USER has explicitly approved this session, recorded on
            `app._approved_write_paths` — approval is a human decision, so an
            approved path is never re-blocked here.
        """
        from ..paths import PathContainmentError, contain_write_path

        healed = self.resolve_path(raw)

        approved = getattr(self.app, "_approved_write_paths", None)
        if approved:
            try:
                if str(Path(healed).resolve()) in {str(Path(a).resolve()) for a in approved}:
                    return Path(healed).resolve()
            except Exception:
                pass

        try:
            return contain_write_path(healed, self.repo)
        except PathContainmentError:
            # Outside the project — the notebook scratch dir is the one
            # sanctioned exception. Checked lazily so the common (contained)
            # case never touches the notebook module.
            try:
                from ..notebook import is_within_notebook
                if is_within_notebook(Path(healed)):
                    return Path(healed).resolve()
            except Exception:
                pass
            try:
                from ..events import emit as _emit
                _emit("write_containment_block", raw=str(raw)[:200],
                      resolved=str(healed)[:200])
            except Exception:
                pass
            raise


@dataclass
class ToolResult:
    """Internal typed tool result.

    Tools may continue returning plain strings. The dispatcher normalizes
    both forms to this type so the agent loop can reason over facts while
    preserving the exact text sent back to the model/UI.
    """
    text: str
    ok: bool = True
    facts: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


class ToolModule(Protocol):
    """Structural type every tool module satisfies."""
    SCHEMA: dict

    @staticmethod
    def execute(ctx: ToolContext, args: dict) -> str: ...  # noqa: E704

    @staticmethod
    def is_concurrency_safe(args: dict) -> bool: ...  # noqa: E704
