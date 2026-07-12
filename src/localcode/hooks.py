"""Lifecycle hooks — user-defined shell commands that run at key events.

Hooks are defined in .localcode/hooks.toml or ~/.localcode/hooks.toml:

    [hooks]
    session_start = "echo 'localcode started' >> /tmp/localcode.log"
    user_prompt_submit = "echo '$PROMPT' >> /tmp/prompts.log"
    pre_tool_use = "if [ '$TOOL_NAME' = 'bash' ]; then echo 'bash: $TOOL_ARGS' >> /tmp/tools.log; fi"
    post_tool_use = ""

Hook types:
    - session_start: runs once when localcode starts
    - user_prompt_submit: runs when user submits a prompt (can modify/block)
    - pre_tool_use: runs before each tool call (can block by returning non-zero)
    - post_tool_use: runs after each tool call (informational)

Environment variables available to hooks:
    $LOCALCODE_SESSION_ID, $LOCALCODE_REPO_ROOT, $LOCALCODE_MODEL
    $PROMPT (for user_prompt_submit)
    $TOOL_NAME, $TOOL_ARGS (for pre/post_tool_use)
    $TOOL_RESULT, $TOOL_ERROR (for post_tool_use)
"""
from __future__ import annotations

import os
import subprocess
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path

from ._subproc_env import clean_env


@dataclass
class HookResult:
    """Result of running a hook."""
    ran: bool = False
    blocked: bool = False       # True if hook returned non-zero (blocks action)
    output: str = ""
    error: str = ""
    exit_code: int = 0


@dataclass
class HookConfig:
    """All configured hooks."""
    session_start: str = ""
    user_prompt_submit: str = ""
    pre_tool_use: str = ""
    post_tool_use: str = ""
    post_edit: str = ""
    pre_completion: str = ""
    post_compaction: str = ""


class HookRunner:
    """Load and execute lifecycle hooks.

    Hooks are shell commands. They receive context via environment variables.
    Pre-hooks can block actions by returning non-zero exit code.
    """

    def __init__(self, repo_root: str, session_id: str = "",
                 model: str = "") -> None:
        self.repo_root = repo_root
        self.session_id = session_id
        self.model = model
        self.config = self._load_hooks()
        self._base_env = {
            **os.environ,
            "LOCALCODE_SESSION_ID": session_id,
            "LOCALCODE_REPO_ROOT": repo_root,
            "LOCALCODE_MODEL": model,
        }

    def _load_hooks(self) -> HookConfig:
        """Load hooks from project .localcode/hooks.toml, then global ~/.localcode/hooks.toml."""
        config = HookConfig()

        # Global hooks (lower priority)
        global_path = Path.home() / ".localcode" / "hooks.toml"
        if global_path.is_file():
            self._merge_hooks(config, global_path)

        # Project hooks (higher priority — overrides global)
        project_path = Path(self.repo_root) / ".localcode" / "hooks.toml"
        if project_path.is_file():
            self._merge_hooks(config, project_path)

        return config

    @staticmethod
    def _merge_hooks(config: HookConfig, path: Path) -> None:
        """Merge hooks from a TOML file into the config."""
        try:
            data = tomllib.loads(path.read_text())
            hooks = data.get("hooks", {})
            for key in (
                "session_start", "user_prompt_submit", "pre_tool_use", "post_tool_use",
                "post_edit", "pre_completion", "post_compaction",
            ):
                val = hooks.get(key, "")
                if val:
                    setattr(config, key, val)
        except Exception:
            pass

    # ── Hook execution ──────────────────────────────────────────────

    def on_session_start(self) -> HookResult:
        """Run session_start hook."""
        if not self.config.session_start:
            return HookResult()
        return self._run(self.config.session_start, {})

    def on_user_prompt_submit(self, prompt: str) -> HookResult:
        """Run user_prompt_submit hook. Non-zero exit = block the prompt."""
        if not self.config.user_prompt_submit:
            return HookResult()
        return self._run(self.config.user_prompt_submit, {"PROMPT": prompt})

    def on_pre_tool_use(self, tool_name: str, tool_args: dict) -> HookResult:
        """Run pre_tool_use hook. Non-zero exit = block the tool call."""
        if not self.config.pre_tool_use:
            return HookResult()
        import json
        return self._run(self.config.pre_tool_use, {
            "TOOL_NAME": tool_name,
            "TOOL_ARGS": json.dumps(tool_args, default=str)[:500],
        })

    def on_post_tool_use(self, tool_name: str, tool_args: dict,
                         result: str, error: bool = False) -> HookResult:
        """Run post_tool_use hook (informational, can't block)."""
        if not self.config.post_tool_use:
            return HookResult()
        import json
        return self._run(self.config.post_tool_use, {
            "TOOL_NAME": tool_name,
            "TOOL_ARGS": json.dumps(tool_args, default=str)[:500],
            "TOOL_RESULT": result[:500],
            "TOOL_ERROR": "1" if error else "0",
        })

    def on_post_edit(self, path: str, result: str, error: bool = False) -> HookResult:
        if not self.config.post_edit:
            return HookResult()
        return self._run(self.config.post_edit, {"EDIT_PATH": path, "EDIT_RESULT": result[:500], "EDIT_ERROR": "1" if error else "0"})

    def on_pre_completion(self, response: str, status: str) -> HookResult:
        if not self.config.pre_completion:
            return HookResult()
        return self._run(self.config.pre_completion, {"COMPLETION_RESPONSE": response[:2000], "COMPLETION_STATUS": status})

    def on_post_compaction(self, before_count: int, after_count: int) -> HookResult:
        if not self.config.post_compaction:
            return HookResult()
        return self._run(self.config.post_compaction, {"COMPACTION_BEFORE_COUNT": str(before_count), "COMPACTION_AFTER_COUNT": str(after_count)})

    def _run(self, command: str, extra_env: dict[str, str]) -> HookResult:
        """Execute a hook command."""
        env = clean_env({**self._base_env, **extra_env})
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.repo_root,
                env=env,
            )
            return HookResult(
                ran=True,
                blocked=(proc.returncode != 0),
                output=proc.stdout.strip()[:500],
                error=proc.stderr.strip()[:500],
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return HookResult(ran=True, blocked=False, error="hook timed out (10s)")
        except Exception as exc:
            return HookResult(ran=True, blocked=False, error=str(exc)[:200])

    @property
    def has_hooks(self) -> bool:
        """Are any hooks configured?"""
        return any([
            self.config.session_start,
            self.config.user_prompt_submit,
            self.config.pre_tool_use,
            self.config.post_tool_use,
            self.config.post_edit,
            self.config.pre_completion,
            self.config.post_compaction,
        ])
