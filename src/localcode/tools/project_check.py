"""Tier-2 verification: run the project's OWN typecheck/lint once, return errors.

The big agents catch semantic errors with a persistent language server (LSP).
opencode's own docs say that for many projects it's better to run the project's
diagnostic CLI directly — LSPs "get out of sync, use significant memory … and
slow down agent workflows." That's decisive for localcode: a 16 GB Mac already
running a large local model can't spare an always-on tsserver/pyright.

So instead we run the project's real typecheck ONCE, when the model is about to
finish an unverified build, and feed the concrete errors back deterministically
(the model can't skip it, and gets ground truth like "line 37: 'getCardsForCard'
does not exist" — the semantic errors a per-write syntax check can't catch).

Light: one bounded subprocess at completion, not per-edit.

Every command run here is READ-ONLY: verification must never write to the user's
repository. That rules out build mode (`tsc -b`), which emits JS/.d.ts/source
maps and always writes `.tsbuildinfo`; referenced projects are type-checked with
`tsc -p <ref> --noEmit` instead, which sees the same errors and emits nothing.
"""
from __future__ import annotations

import json
import os
import re
import hashlib
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass

_TIMEOUT = 60.0
_MAX_ERR_CHARS = 2500  # floor; scaled up to the real context window at call time
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# O_DIRECTORY exists on POSIX; 0 is a harmless no-op elsewhere.
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_TSCONFIG_MAX_BYTES = 512_000
_MAX_TS_PROJECTS = 12   # projects type-checked per run
_MAX_TS_NODES = 40      # tsconfig files visited while resolving the graph
_MAX_TS_DEPTH = 8       # reference-graph depth walked


@dataclass(frozen=True)
class CheckOutcome:
    """The result of the completion-gate verification.

    `status` is deliberately explicit because "no error text" USED to mean five
    different things at once — clean, no checker installed, timed out, failed to
    execute, or failed with no output — and the gate read all five as CLEAN. A
    false clean is the one failure mode this gate exists to prevent.

      clean        the checker ran and the project is green
      errors       the checker ran and reported diagnostics (see `detail`)
      unavailable  nothing to run (no checker installed) — the gate is unverified
      timed_out    the checker exceeded the timeout — unverified, NOT clean
      failed       could not run, or exited nonzero with no usable output
    """

    status: str
    label: str = ""
    detail: str = ""

    @property
    def is_red(self) -> bool:
        return self.status == "errors"

    @property
    def is_verified(self) -> bool:
        """True only when a checker actually ran to a green result."""
        return self.status == "clean"


# Ranked worst-first: a run where one checker is green and another timed out is
# reported as timed_out, never as clean.
_STATUS_RANK = {"unavailable": 0, "clean": 1, "timed_out": 2, "failed": 3, "errors": 4}


def _strip_jsonc(text: str) -> str:
    """Remove `//` and `/* */` comments and trailing commas, preserving string
    literals exactly. TypeScript has officially supported comments in
    `tsconfig.json` since 1.8 and every scaffold ships them, so a strict
    `json.load()` fails on the majority of real solution configs.

    Trailing commas are removed inside the SAME stateful scan, never by a regex
    over the whole text: a comma-then-bracket regex is not string-aware and
    silently rewrites a path literal such as `"./bad,}"` into
    `"./bad}"`, which points reference resolution at a file that does not exist
    and turns a real error into a false clean.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False

    def _drop_pending_comma() -> None:
        """About to emit `}` or `]` outside a string: remove a comma that is now
        trailing. Only whitespace may sit between it and the bracket, and `out`
        holds no comments at this point, so the lookback cannot cross a literal."""
        j = len(out) - 1
        while j >= 0 and out[j] in " \t\r\n":
            j -= 1
        if j >= 0 and out[j] == ",":
            del out[j]

    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        if c in "}]":
            _drop_pending_comma()
        out.append(c)
        i += 1
    return "".join(out)


def _load_jsonc(path: str) -> tuple[dict | None, str]:
    """Parse a JSONC config. Returns `(data, "")` on success or `(None, reason)`.

    A parse failure NEVER falls back to a weaker command: the caller reports the
    verification as unavailable, because silently downgrading to a check that is
    known not to cover the project is exactly how a false clean gets through.
    """
    try:
        if os.path.getsize(path) > _TSCONFIG_MAX_BYTES:
            return None, f"{os.path.basename(path)} is too large to parse"
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except Exception as exc:
        return None, f"cannot read {os.path.basename(path)}: {exc.__class__.__name__}"
    try:
        data = json.loads(_strip_jsonc(raw))
    except Exception as exc:
        return None, f"cannot parse {os.path.basename(path)}: {exc.__class__.__name__}"
    if not isinstance(data, dict):
        return None, f"{os.path.basename(path)} is not a JSON object"
    return data, ""


def run_project_check(repo_root: str, ctx_tokens: int = 0) -> str | None:
    """Back-compat wrapper: bounded error text when the project is RED, else
    None. New callers should use `run_project_check_result`, which distinguishes
    clean from unavailable/timed-out/failed."""
    outcome = run_project_check_result(repo_root, ctx_tokens)
    return outcome.detail if outcome.is_red else None


def run_project_check_result(repo_root: str, ctx_tokens: int = 0) -> CheckOutcome:
    """Run the first available typecheck/lint for the project and report a
    structured outcome.

    `ctx_tokens` (the model's real context window) scales the error-text cap so a
    big-window model can see more of a wall of tsc/build errors it has room for,
    instead of a fixed 2500-char slice; floored at 2500 so small machines are
    unchanged.
    """
    max_err_chars = _MAX_ERR_CHARS
    if ctx_tokens:
        max_err_chars = max(_MAX_ERR_CHARS, int(ctx_tokens * 3.5 * 0.04))
    env = {**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"}
    worst = CheckOutcome("unavailable", "", "no project checker available")

    def _keep(candidate: CheckOutcome) -> CheckOutcome:
        return candidate if _STATUS_RANK[candidate.status] > _STATUS_RANK[worst.status] else worst

    for label, argv, cwd in _detect_commands(repo_root):
        if argv is None:
            # The detector found a checker but could not determine a SAFE,
            # adequate command (unparseable tsconfig, emitting build config).
            # Unverified — never a silent downgrade to a weaker check.
            worst = _keep(CheckOutcome("failed", label, label))
            continue
        try:
            r = subprocess.run(
                argv, cwd=cwd, capture_output=True, text=True,
                timeout=_TIMEOUT, env=env,
            )
        except subprocess.TimeoutExpired:
            worst = _keep(CheckOutcome(
                "timed_out", label, f"[{label}] timed out after {_TIMEOUT:.0f}s"))
            continue
        except Exception as exc:
            worst = _keep(CheckOutcome(
                "failed", label, f"[{label}] could not run: {exc.__class__.__name__}"))
            continue
        if r.returncode != 0:
            out = _ANSI.sub("", ((r.stdout or "") + "\n" + (r.stderr or "")).strip())
            # Keep only the most useful lines, capped, so we don't flood a
            # small model's context with a wall of output. The line cap also
            # scales with the window (40 on a small machine → more on a big one).
            max_lines = 40 if not ctx_tokens else max(40, int(ctx_tokens / 4000))
            lines = [l for l in out.splitlines() if l.strip()][:max_lines]
            if any("TS6304" in l for l in lines):
                # "Composite projects may not disable emit" — a toolchain
                # limitation (older TypeScript), not a defect in the user's code.
                # Report unverified rather than feeding a bogus error back.
                worst = _keep(CheckOutcome(
                    "failed", label,
                    f"[{label}] cannot type-check without emitting (TypeScript too old)"))
                continue
            if lines:
                detail = f"[{label}] reported errors:\n" + "\n".join(lines)[:max_err_chars]
                return CheckOutcome("errors", label, detail)
            # Nonzero with nothing to show: the checker failed, the project is
            # NOT proven clean. Reporting this as clean is a false clean.
            worst = _keep(CheckOutcome(
                "failed", label,
                f"[{label}] exited {r.returncode} with no output"))
            continue
        worst = _keep(CheckOutcome("clean", label, ""))
    return worst


def _detect_commands(repo_root: str) -> list[tuple[str, list[str] | None, str]]:
    """Ordered (label, argv, cwd) checkers to try.

    An argv of `None` is a deliberate sentinel meaning "a checker exists for this
    project but no SAFE, adequate command could be determined" (e.g. an
    unparseable tsconfig). The caller reports that as a FAILED verification; it
    must never be downgraded to a command known not to cover the project.
    """
    cmds: list[tuple[str, list[str] | None, str]] = []

    # ── JS / TS ── prefer a project "typecheck" script, else tsc --noEmit.
    pj_dir = _nearest_with(repo_root, "package.json")
    if pj_dir:
        node_modules = os.path.join(pj_dir, "node_modules")
        binp = os.path.join(node_modules, ".bin")
        scripts = {}
        try:
            scripts = (json.load(open(os.path.join(pj_dir, "package.json"))) or {}).get("scripts", {})
        except Exception:
            scripts = {}
        # Only run node-based checks once deps are installed (else every import
        # is a false "cannot find module" error that would derail the model).
        # NOTE: tsc/typecheck only covers the files the project's tsconfig
        # includes (typically `src/`). A module written OUTSIDE that root (e.g.
        # accidentally at the repo root) is never type-checked here, so the
        # Tier-1 per-write syntax_check is the only gate it passes through. If
        # this proves a recurring miss, widen coverage (e.g. an explicit tsc
        # over stray *.ts outside `include`) rather than assuming src/-only.
        if os.path.isdir(node_modules):
            # SECURITY: this verification runs UNATTENDED (auto-invoked at the
            # build_app completion gate, no approval). Do NOT run
            # `npm run <script>` here — a script is an arbitrary shell string
            # from the repo's package.json, so a malicious/injected repo would
            # get code execution the moment the agent verifies. Run the real
            # type-checker binary directly instead: `tsc --noEmit` reads only
            # tsconfig.json (pure JSON/data) and never executes project code.
            # eslint is deliberately NOT auto-run — its config (.eslintrc.js /
            # eslint.config.js) executes JS, which is the same RCE vector.
            if os.path.exists(os.path.join(pj_dir, "tsconfig.json")) and os.path.exists(os.path.join(binp, "tsc")):
                tsc = os.path.join(binp, "tsc")
                # Modern Vite/React/Vue scaffolds use TS PROJECT REFERENCES: the
                # root tsconfig.json is a solution file (`"references": [...]`,
                # no `include`), so `tsc --noEmit` on it type-checks NOTHING and
                # reports a FALSE clean — the hole that let a build with 3 real
                # tsc errors complete as "verified". Type-check each REFERENCED
                # project directly with `tsc -p <ref> --noEmit`.
                #
                # NOT `tsc -b`: build mode is a BUILD, not a typecheck. It emits
                # JS, declarations and source maps into the user's repo (since TS
                # 5.6 on a best-effort basis even when a project has errors) and
                # always writes `.tsbuildinfo` — a verification step silently
                # mutating the user's files, unattended. `-p … --noEmit` surfaces
                # the same diagnostics and writes nothing at all.
                targets, reason = _ts_check_targets(pj_dir)
                # `composite: true` implies `incremental`, so even in --noEmit
                # mode tsc insists on writing a .tsbuildinfo. It goes OUTSIDE the
                # repo, and if no such directory can be verified we do NOT run —
                # running without the redirect writes into the user's project.
                scratch = _tsbuildinfo_dir(pj_dir) if targets else None
                if targets and not scratch:
                    targets, reason = [], (
                        "no writable scratch directory outside the repository, so "
                        "type-checking cannot be run without writing to your project")
                for cfg in targets:
                    argv = [tsc, "-p", cfg, "--noEmit", "--pretty", "false",
                            "--tsBuildInfoFile",
                            os.path.join(scratch, f"{cfg.replace(os.sep, '_')}.tsbuildinfo")]
                    cmds.append((f"tsc -p {cfg} --noEmit", argv, pj_dir))
                if reason:
                    # Emitted ALONGSIDE any commands above: partial coverage must
                    # not read as clean, but real diagnostics still outrank it.
                    cmds.append((f"tsc (verification incomplete: {reason})", None, pj_dir))

    # ── Python ── ruff for REAL errors only (E9 syntax + F pyflakes: undefined
    # names, bad imports) — not style noise. Falls back to compileall (syntax).
    if _has_ext(repo_root, ".py"):
        if shutil.which("ruff"):
            cmds.append(("ruff", ["ruff", "check", "--select", "E9,F",
                                  "--no-cache", "--output-format", "concise", "."], repo_root))
        elif shutil.which("python3") or shutil.which("python"):
            py = shutil.which("python3") or shutil.which("python")
            cmds.append(("python -m compileall", [py, "-m", "compileall", "-q", "."], repo_root))

    # ── Go ── the compiler catches type/undefined errors (like tsc). go.mod.
    go_dir = _nearest_with(repo_root, "go.mod")
    if go_dir and shutil.which("go"):
        cmds.append(("go build", ["go", "build", "./..."], go_dir))

    # ── Rust ── cargo check is the type-checker (fast, no codegen). Cargo.toml.
    rust_dir = _nearest_with(repo_root, "Cargo.toml")
    if rust_dir and shutil.which("cargo"):
        cmds.append(("cargo check", ["cargo", "check", "--message-format", "short"], rust_dir))

    # (Interpreted langs — Ruby/PHP/shell — have no whole-project type-check;
    # their per-file syntax linters run in the Tier-1 syntax_check on each write.)
    return cmds


def _dir_is_inside(fd: int, repo_id: tuple[int, int]) -> bool:
    """Is the directory held by `fd` inside the repository?

    Walks `..` upward using directory FILE DESCRIPTORS and compares
    `(st_dev, st_ino)` — never paths. An open fd names an inode, so nothing an
    attacker renames or symlinks afterwards can change what it refers to, and
    there is no string for a symlink to redirect.
    """
    cur = os.dup(fd)
    try:
        while True:
            st = os.fstat(cur)
            if (st.st_dev, st.st_ino) == repo_id:
                return True
            parent = os.open("..", os.O_RDONLY | _O_DIRECTORY, dir_fd=cur)
            try:
                pst = os.fstat(parent)
                if (pst.st_dev, pst.st_ino) == (st.st_dev, st.st_ino):
                    return False          # `..` of the root is the root
            except Exception:
                os.close(parent)
                raise
            os.close(cur)
            cur = parent
    finally:
        try:
            os.close(cur)
        except Exception:
            pass


def _path_of_fd(fd: int) -> str:
    """The kernel's own fully-resolved path for an open directory, or "".

    Needed because the caller must hand tsc a PATH: composing one from the
    original string would send tsc back through the very symlink the fd checks
    bypassed. This asks the kernel where the fd actually points, so the result
    contains no symlinked components at all.
    """
    try:
        if sys.platform == "linux":
            return os.readlink(f"/proc/self/fd/{fd}")
        import fcntl
        _F_GETPATH = 50  # macOS / BSD
        buf = fcntl.fcntl(fd, _F_GETPATH, b"\0" * 1024)
        return buf.split(b"\0", 1)[0].decode("utf-8", "replace")
    except Exception:
        return ""


def _tsbuildinfo_dir(pj_dir: str) -> str | None:
    """A scratch directory OUTSIDE the repo for tsc's incremental state, or None.

    `composite: true` implies `incremental`, so tsc writes a `.tsbuildinfo` even
    under `--noEmit`. None means the verification MUST NOT run: running without
    `--tsBuildInfoFile` puts that file back inside the user's repository while
    still reporting clean.

    Creation is anchored to an OPEN DIRECTORY HANDLE, not to a path. Validating a
    path and then calling `os.makedirs` on it is a TOCTOU: swapping the validated
    prefix to a symlink in the window between the two created the directory
    inside the repository, and a post-creation check could only notice the damage
    afterwards. Here the temp directory is opened once, its containment is proven
    from that fd, and the child is created with `os.mkdir(name, dir_fd=fd)` — a
    single component resolved against the pinned inode, which a later swap of the
    path prefix cannot redirect. `mkdir` never follows a symlink for the final
    component either, so the child cannot be aimed elsewhere.

    The path handed back is the KERNEL's path for the created directory, not the
    string we started from — otherwise tsc would be sent back through the same
    symlink the fd work just bypassed. If that path cannot be obtained or no
    longer resolves to the directory we created, this returns None and the check
    refuses to run.

    Residual, stated plainly: tsc is a subprocess and takes a path, so an
    attacker who can replace a REAL directory component of the resolved temp
    path with a symlink after this returns could still redirect tsc's own write.
    Closing that entirely would need tsc to accept a file descriptor, which it
    does not. What is guaranteed is that LocalCode creates nothing inside the
    repository and never hands out a path containing an attacker-supplied
    symlink.
    """
    key = hashlib.sha1(os.path.abspath(pj_dir).encode("utf-8", "replace")).hexdigest()[:16]
    name = f"localcode-tscheck-{key}"
    parent = tempfile.gettempdir()
    fd = None
    child_fd = None
    try:
        repo_st = os.stat(pj_dir)
        repo_id = (repo_st.st_dev, repo_st.st_ino)
        # Opening resolves any symlink in `parent` ONCE; every check below is
        # against the resulting inode, so a later swap has nothing to act on.
        fd = os.open(parent, os.O_RDONLY | _O_DIRECTORY)
        if _dir_is_inside(fd, repo_id):
            return None
        try:
            os.mkdir(name, dir_fd=fd)
        except FileExistsError:
            pass
        # Reject a pre-existing symlink squatting on the name: `mkdir` would have
        # raised EEXIST for it, and following it is how the write escapes.
        st = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if not stat.S_ISDIR(st.st_mode):
            return None
        child_fd = os.open(name, os.O_RDONLY | _O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
        if _dir_is_inside(child_fd, repo_id):
            return None
        # Prove it is writable, again relative to the pinned handle.
        probe = os.open(".probe", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                        dir_fd=child_fd)
        os.close(probe)
        os.unlink(".probe", dir_fd=child_fd)
        resolved = _path_of_fd(child_fd)
        if not resolved or not os.path.isabs(resolved):
            return None
        # The kernel path must still name the directory we just created, and
        # must still sit outside the repository.
        st_named = os.stat(resolved)
        st_child = os.fstat(child_fd)
        if (st_named.st_dev, st_named.st_ino) != (st_child.st_dev, st_child.st_ino):
            return None
        repo_real = os.path.realpath(pj_dir)
        if resolved == repo_real or os.path.commonpath([resolved, repo_real]) == repo_real:
            return None
        return resolved
    except Exception:
        return None
    finally:
        for handle in (child_fd, fd):
            if handle is not None:
                try:
                    os.close(handle)
                except Exception:
                    pass


def _ts_check_targets(pj_dir: str) -> tuple[list[str], str]:
    """Config files to type-check for a TS project, relative to `pj_dir`, plus a
    reason string when coverage is INCOMPLETE.

    Both halves matter. `(targets, "")` is full coverage. A non-empty reason
    means the graph could not be fully resolved or had to be truncated, and the
    caller MUST report the run as unverified even when the targets it did check
    came back green — silently doing less work is exactly how a false clean gets
    through.

    Specifics the previous version got wrong, each proven against real tsc:
      * a config with BOTH `references` and its own `include`/`files` OWNS
        source and must be checked itself, not treated as a pure solution file;
      * an unresolvable reference is an error (`tsc -b` reports TS5083), never
        something to skip because a sibling reference happened to resolve;
      * truncation — of targets, of visited nodes, or of depth — is reported,
        not swallowed. Depth and node count are bounded, not just the target
        list.
    """
    root = os.path.join(pj_dir, "tsconfig.json")
    data, err = _load_jsonc(root)
    if data is None:
        return [], err

    def _refs(cfg: dict, cfg_path: str) -> tuple[list[str], str]:
        """Reference `path` values, or a reason when the list is malformed."""
        raw = cfg.get("references")
        if raw is None:
            return [], ""
        if not isinstance(raw, list):
            return [], f"{os.path.basename(cfg_path)} has a non-list `references`"
        out = []
        for entry in raw:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                return [], f"{os.path.basename(cfg_path)} has an invalid reference entry"
            out.append(entry["path"])
        return out, ""

    def _owns_source(cfg: dict) -> bool:
        """True when the config contributes files of its own.

        A solution file opts out explicitly with `"files": []` / `"include": []`.
        With NEITHER key present TypeScript defaults to including everything, so
        the config does own source and must be checked.
        """
        files, include = cfg.get("files"), cfg.get("include")
        if isinstance(files, list) and files:
            return True
        if isinstance(include, list) and include:
            return True
        explicit_empty = (isinstance(files, list) and not files) or (
            isinstance(include, list) and not include)
        return not explicit_empty

    targets: list[str] = []
    visited: set[str] = {os.path.realpath(root)}
    queue: list[tuple[dict, str, int]] = [(data, root, 0)]
    truncated = ""

    def _add_target(cfg_path: str) -> None:
        nonlocal truncated
        rel = os.path.relpath(cfg_path, pj_dir)
        if rel in targets:
            return
        if len(targets) >= _MAX_TS_PROJECTS:
            truncated = truncated or (
                f"more than {_MAX_TS_PROJECTS} TypeScript projects — "
                f"only the first {_MAX_TS_PROJECTS} were type-checked")
            return
        targets.append(rel)

    while queue:
        cfg, cfg_path, depth = queue.pop(0)
        refs, ref_err = _refs(cfg, cfg_path)
        if ref_err:
            return [], ref_err
        if _owns_source(cfg):
            _add_target(cfg_path)
        if not refs:
            continue
        if depth >= _MAX_TS_DEPTH:
            truncated = truncated or (
                f"reference graph deeper than {_MAX_TS_DEPTH} levels — "
                f"the deepest projects were not type-checked")
            continue
        for ref in refs:
            # `path` may point at a config file or at a directory containing one.
            cand = os.path.normpath(os.path.join(os.path.dirname(cfg_path), ref))
            if os.path.isdir(cand):
                cand = os.path.join(cand, "tsconfig.json")
            if not os.path.isfile(cand):
                # `tsc -b` fails with TS5083 here; skipping it would hide every
                # error in the project that reference points at.
                return [], f"reference not found: {os.path.relpath(cand, pj_dir)}"
            real = os.path.realpath(cand)
            if real in visited:
                continue
            if len(visited) >= _MAX_TS_NODES:
                truncated = truncated or (
                    f"reference graph larger than {_MAX_TS_NODES} configs — "
                    f"it was not fully resolved")
                break
            visited.add(real)
            sub, sub_err = _load_jsonc(cand)
            if sub is None:
                return [], sub_err
            queue.append((sub, cand, depth + 1))

    if not targets:
        # A truncation reason is more specific than "nothing owns source", and
        # is what actually explains the missing coverage.
        return [], truncated or "tsconfig.json resolves to no project that owns source files"
    return targets, truncated


def _nearest_with(root: str, filename: str) -> str | None:
    """The shallowest directory under root that contains `filename`."""
    root = os.path.abspath(root)
    if os.path.exists(os.path.join(root, filename)):
        return root
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "dist", "build", ".venv")]
        if filename in files:
            return base
    return None


def _has_ext(root: str, ext: str) -> bool:
    for base, dirs, files in os.walk(os.path.abspath(root)):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "dist", "build", ".venv")]
        if any(f.endswith(ext) for f in files):
            return True
    return False
