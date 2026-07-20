"""The build target-location directive must NOT anchor to $HOME.

When localcode is launched outside any project (no marker found), repo_root
falls back to the user's home dir. Injecting "the project root is /Users/x,
create all files under that root" then makes the model build in $HOME or
conflict with the path the user named in the task.
"""
from pathlib import Path

from localcode.agent.prompt_context import build_target_grounding_block


class _Goal:
    goal_type = "build_app"


def test_home_root_emits_no_directive():
    assert build_target_grounding_block(str(Path.home()), _Goal()) == ""


def test_real_project_root_emits_directive():
    block = build_target_grounding_block(
        "/Users/x/Desktop/Github/localcode_test/Anki/myapp", _Goal()
    )
    assert "work here" in block
    assert "myapp" in block


def test_non_build_goal_emits_nothing():
    class G:
        goal_type = "answer"
    assert build_target_grounding_block("/some/project", G()) == ""
