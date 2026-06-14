"""Tests for the checkpoint-and-branch context-control module.

Pure logic only — no models, no network. Filesystem use is confined to tmp_path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localcode.checkpoint import (
    Checkpoint,
    CheckpointStore,
    FileState,
    RevertPlan,
    restore_files,
)


def _msgs(*texts: str) -> list[dict]:
    return [{"role": "user", "content": t} for t in texts]


# ---------------------------------------------------------------- create/list
def test_create_and_get_and_list():
    store = CheckpointStore()
    a = store.create("milestone-a", _msgs("hello"))
    b = store.create("milestone-b", _msgs("hello", "world"))

    assert store.get(a).label == "milestone-a"
    assert store.get(b).label == "milestone-b"
    labels = [c.label for c in store.list()]
    assert labels == ["milestone-a", "milestone-b"]  # oldest first


def test_get_unknown_returns_none():
    store = CheckpointStore()
    assert store.get("nope") is None


def test_get_prefix_match():
    store = CheckpointStore()
    cid = store.create("m", _msgs("x"))
    assert store.get(cid[:4]).id == cid


def test_messages_are_deep_copied_on_create():
    store = CheckpointStore()
    msgs = _msgs("original")
    cid = store.create("m", msgs)
    msgs[0]["content"] = "MUTATED"  # mutate live conversation after checkpoint
    assert store.get(cid).messages[0]["content"] == "original"


# ----------------------------------------------------------------- file state
def test_create_captures_file_content(tmp_path: Path):
    (tmp_path / "f.txt").write_text("v1")
    store = CheckpointStore(repo_root=tmp_path)
    cid = store.create("m", _msgs("x"), changed_files=["f.txt"])
    cp = store.get(cid)
    assert cp.changed_files == ["f.txt"]
    fs = cp.files["f.txt"]
    assert fs.existed and fs.content == "v1" and fs.content_hash


def test_create_records_nonexistent_file(tmp_path: Path):
    store = CheckpointStore(repo_root=tmp_path)
    cid = store.create("m", _msgs("x"), changed_files=["new.txt"])
    fs = store.get(cid).files["new.txt"]
    assert not fs.existed and fs.content == ""


def test_capture_content_false_drops_body_keeps_hash(tmp_path: Path):
    (tmp_path / "f.txt").write_text("body")
    store = CheckpointStore(repo_root=tmp_path)
    cid = store.create("m", _msgs("x"), changed_files=["f.txt"],
                       capture_content=False)
    fs = store.get(cid).files["f.txt"]
    assert fs.existed and fs.content == "" and fs.content_hash


# --------------------------------------------------------------------- revert
def test_revert_returns_right_messages_and_files(tmp_path: Path):
    (tmp_path / "f.txt").write_text("checkpoint-content")
    store = CheckpointStore(repo_root=tmp_path)
    cid = store.create("m", _msgs("a", "b"), changed_files=["f.txt"])

    plan = store.revert(cid)
    assert isinstance(plan, RevertPlan)
    assert [m["content"] for m in plan.messages] == ["a", "b"]
    assert len(plan.files_to_restore) == 1
    assert plan.files_to_restore[0].content == "checkpoint-content"


def test_revert_is_pure_no_side_effects(tmp_path: Path):
    f = tmp_path / "f.txt"
    f.write_text("v1")
    store = CheckpointStore(repo_root=tmp_path)
    cid = store.create("m", _msgs("x"), changed_files=["f.txt"])
    f.write_text("v2-changed-later")

    store.revert(cid)  # should NOT touch disk
    assert f.read_text() == "v2-changed-later"


def test_revert_unknown_raises():
    store = CheckpointStore()
    with pytest.raises(KeyError):
        store.revert("missing")


def test_revert_plan_messages_are_isolated():
    store = CheckpointStore()
    cid = store.create("m", _msgs("orig"))
    plan = store.revert(cid)
    plan.messages[0]["content"] = "edited"
    assert store.get(cid).messages[0]["content"] == "orig"


# ------------------------------------------------------------- restore_files
def test_restore_files_writes_and_deletes(tmp_path: Path):
    f = tmp_path / "f.txt"
    f.write_text("good")
    store = CheckpointStore(repo_root=tmp_path)
    # checkpoint: f.txt exists w/ "good", new.txt does not exist
    cid = store.create("m", _msgs("x"), changed_files=["f.txt", "new.txt"])

    # simulate later work: corrupt f.txt, create new.txt
    f.write_text("BAD")
    (tmp_path / "new.txt").write_text("should-be-deleted")

    plan = store.revert(cid)
    report = restore_files(plan, tmp_path)

    assert f.read_text() == "good"
    assert not (tmp_path / "new.txt").exists()
    assert report["f.txt"] == "restored"
    assert report["new.txt"] == "deleted"


def test_revert_plan_write_delete_partition(tmp_path: Path):
    (tmp_path / "exists.txt").write_text("e")
    store = CheckpointStore(repo_root=tmp_path)
    cid = store.create("m", _msgs("x"),
                       changed_files=["exists.txt", "gone.txt"])
    plan = store.revert(cid)
    assert [f.path for f in plan.files_to_write] == ["exists.txt"]
    assert [f.path for f in plan.files_to_delete] == ["gone.txt"]


# ------------------------------------------------------------------ branching
def test_branch_from_parent_sets_parent_id():
    store = CheckpointStore()
    root = store.create("root", _msgs("a"))
    child = store.create("approach-b", _msgs("a", "b"), parent=root)
    cp = store.get(child)
    assert cp.parent_id == root


def test_create_with_unknown_parent_raises():
    store = CheckpointStore()
    with pytest.raises(ValueError):
        store.create("x", _msgs("a"), parent="nope")


def test_children_and_siblings():
    store = CheckpointStore()
    root = store.create("root", _msgs("a"))
    b1 = store.create("approach-1", _msgs("a", "b1"), parent=root)
    b2 = store.create("approach-2", _msgs("a", "b2"), parent=root)
    child_ids = [c.id for c in store.children(root)]
    assert child_ids == [b1, b2]  # two alternative branches off the milestone
    assert store.children(b1) == []


def test_branch_helper_creates_child_and_revert_plan(tmp_path: Path):
    (tmp_path / "f.txt").write_text("milestone")
    store = CheckpointStore(repo_root=tmp_path)
    milestone = store.create("milestone", _msgs("a", "b"),
                             changed_files=["f.txt"])

    plan, child_id = store.branch(milestone, "try-different-approach")
    assert [m["content"] for m in plan.messages] == ["a", "b"]
    assert plan.files_to_restore[0].content == "milestone"
    assert store.get(child_id).parent_id == milestone
    assert store.get(child_id).label == "try-different-approach"


def test_branch_unknown_raises():
    store = CheckpointStore()
    with pytest.raises(KeyError):
        store.branch("nope", "x")


def test_lineage_root_to_leaf():
    store = CheckpointStore()
    a = store.create("a", _msgs("1"))
    b = store.create("b", _msgs("1", "2"), parent=a)
    c = store.create("c", _msgs("1", "2", "3"), parent=b)
    assert [cp.label for cp in store.lineage(c)] == ["a", "b", "c"]


# ---------------------------------------------------------------- persistence
def test_json_round_trip(tmp_path: Path):
    (tmp_path / "f.txt").write_text("content-v1")
    store = CheckpointStore(repo_root=tmp_path)
    root = store.create("root", _msgs("a"), changed_files=["f.txt"])
    child = store.create("child", _msgs("a", "b"), parent=root)

    out = store.save(tmp_path / "ckpt.json")
    loaded = CheckpointStore.load(out)

    assert {c.id for c in loaded.list()} == {root, child}
    rcp = loaded.get(root)
    assert rcp.label == "root"
    assert rcp.files["f.txt"].content == "content-v1"
    assert loaded.get(child).parent_id == root
    # revert still works after reload
    plan = loaded.revert(child)
    assert [m["content"] for m in plan.messages] == ["a", "b"]


def test_persist_dir_autosaves(tmp_path: Path):
    pdir = tmp_path / ".localcode"
    store = CheckpointStore(repo_root=tmp_path, persist_dir=pdir)
    store.create("auto", _msgs("x"))
    assert (pdir / "checkpoints.json").is_file()


def test_filestate_dict_round_trip():
    fs = FileState(path="p", existed=True, content="c", content_hash="h")
    assert FileState.from_dict(fs.to_dict()) == fs


def test_checkpoint_dict_round_trip():
    cp = Checkpoint(id="i", label="L", messages=_msgs("m"),
                    files={"p": FileState("p", True, "c", "h")},
                    round_marker=3, token_marker=42, parent_id=None)
    back = Checkpoint.from_dict(cp.to_dict())
    assert back.id == cp.id and back.round_marker == 3 and back.token_marker == 42
    assert back.files["p"].content == "c"
