"""Patch-created files are removed only when they match the manifest."""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("state", ["generated", "modified", "staged-modified", "symlink"])
def test_build_cleanup_only_removes_exact_generated_files(tmp_path, state):
    script = (Path(__file__).parents[1] / "scripts/build_ui_binary.sh").read_text()
    cleanup = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    repo, patches = tmp_path / "repo", tmp_path / "patches"
    repo.mkdir()
    patches.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    relative = "packages/tui/src/ui/generated.ts"
    (patches / "change.patch").write_bytes((
        f"diff --git a/{relative} b/{relative}\nnew file mode 100644\n"
        f"--- /dev/null\n+++ b/{relative}\n@@ -0,0 +1 @@\n+export const value = 1\n"
    ).encode())
    target = repo / relative
    target.parent.mkdir(parents=True)
    outside = tmp_path / "private.ts"
    outside.write_text("private edits\n")
    if state == "symlink":
        target.symlink_to(outside)
    else:
        target.write_bytes(b"export const value = 1\n" if state == "generated" else b"private edits\n")
        if state == "staged-modified":
            subprocess.run(["git", "add", "-N", relative], cwd=repo, check=True)
    result = subprocess.run([sys.executable, "-c", cleanup, str(patches), str(repo)], capture_output=True)
    if state == "generated":
        assert result.returncode == 0 and not target.exists()
    else:
        assert result.returncode != 0
        assert target.read_text() == "private edits\n"
    assert outside.read_text() == "private edits\n"
