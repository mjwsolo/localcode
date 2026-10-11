"""The sandbox each sample works in: a plain directory with a readable name.

Inspect's local sandbox puts a sample in the system temp directory, which on
macOS is a path like /var/folders/6m/cd6dvv2519l0ts64fdxrrq180000gn/T/tmpe3fw3any.
Small models miscopy that string (North Mini Code wrote `cd6dvv2519-l0ts...` in
three of every four file paths), the miscopied path lies outside the workspace,
the headless UI refuses it, and the turn ends. That measures how well a model
copies thirty random characters, which no real project path asks of it.

This sandbox is the local one with the directory named for the sample:
<root>/<sample>-<n>, root being LOCALCODE_EVALS_WORKROOT or /tmp/localcode-evals.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from inspect_ai.util import SandboxEnvironment, SandboxEnvironmentConfigType, sandboxenv
from inspect_ai.util._sandbox.local import LocalSandboxEnvironment


class _Directory:
    """What the local sandbox expects of its directory: a name and a cleanup."""

    def __init__(self, label: str) -> None:
        root = Path(os.environ.get("LOCALCODE_EVALS_WORKROOT") or "/tmp/localcode-evals").resolve()
        root.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40] or "sample"
        n = 1
        while True:
            try:
                (root / f"{slug}-{n}").mkdir()
                break
            except FileExistsError:
                n += 1
        self.name = str(root / f"{slug}-{n}")

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)


@sandboxenv(name="workspace")
class WorkspaceSandbox(LocalSandboxEnvironment):
    @classmethod
    async def sample_init(cls, task_name: str, config: SandboxEnvironmentConfigType | None,
                          metadata: dict[str, str]) -> dict[str, SandboxEnvironment]:
        return {"default": cls(str(metadata.get("id") or metadata.get("title") or task_name))}

    def __init__(self, label: str = "sample") -> None:
        super().__init__()
        self.directory.cleanup()                     # the temp directory the base class made
        self.directory = _Directory(label)           # type: ignore[assignment]
        self._sandbox_tools_dir = Path(self.directory.name) / "sandbox-tools"
