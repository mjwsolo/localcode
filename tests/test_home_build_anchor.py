"""Regression: a $HOME-launched new-app build must anchor to a real project dir.

Half of the Anki-build stop was that the build ran with repo_root=$HOME — the
model scattered files while launcher/verify/context stayed pinned at $HOME.
`_anchor_home_build_root` re-pins to $HOME/<slug> before the first tool call.
Tested in isolation with a lightweight stub so we don't spin up a real model.
"""
import logging
import types
from pathlib import Path

from localcode.app import LocalCodeApp


def _stub(home: Path):
    changes = types.SimpleNamespace(repo_root=home)
    toolkit = types.SimpleNamespace(repo_root=home, changes=changes)
    return types.SimpleNamespace(
        repo_root=home,
        repo_root_is_home=True,
        log=logging.getLogger("test"),
        toolkit=toolkit,
    )


def test_anchor_creates_and_repins_to_canonical_dir(tmp_path: Path) -> None:
    fake = _stub(tmp_path)
    had_files = LocalCodeApp._anchor_home_build_root(fake, "local-first-anki-clone")
    expected = (tmp_path / "local-first-anki-clone").resolve()
    assert had_files is False
    assert expected.is_dir()
    # repo_root and the path-critical components all follow the new root.
    assert fake.repo_root == expected
    assert fake.repo_root_is_home is False
    assert fake.toolkit.repo_root == expected
    assert fake.toolkit.changes.repo_root == expected


def test_anchor_is_idempotent_and_flags_resume(tmp_path: Path) -> None:
    # An existing canonical dir with files is a resume, not a fresh build.
    proj = tmp_path / "my-app"
    proj.mkdir()
    (proj / "package.json").write_text("{}\n")
    fake = _stub(tmp_path)
    assert LocalCodeApp._anchor_home_build_root(fake, "my-app") is True
    assert fake.repo_root == proj.resolve()


def test_anchor_rejects_empty_slug_and_traversal(tmp_path: Path) -> None:
    fake = _stub(tmp_path)
    assert LocalCodeApp._anchor_home_build_root(fake, "") is False
    assert fake.repo_root == tmp_path  # unchanged
    assert fake.repo_root_is_home is True
    # A slug that tries to climb out of $HOME must be refused.
    fake2 = _stub(tmp_path)
    LocalCodeApp._anchor_home_build_root(fake2, "..")
    assert fake2.repo_root == tmp_path
