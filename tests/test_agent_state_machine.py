from localcode.agent.goal import infer_goal_state
from localcode.agent.state_machine import TaskEvent, TaskStage, event_for_tool, transition


def test_conservative_goal_classifier_activates_harness_lifecycles() -> None:
    assert infer_goal_state("build a small dashboard app").goal_type == "build_app"
    assert infer_goal_state("fix the import bug").goal_type == "edit_existing"
    assert infer_goal_state("run the app").goal_type == "run_or_launch"
    assert infer_goal_state("why does this fail?").goal_type == "question"
    assert infer_goal_state("organize these notes").goal_type == "general_task"


def test_task_lifecycle_requires_verification_before_completion() -> None:
    assert transition("planning", TaskEvent.MUTATION_SUCCEEDED).after == TaskStage.IMPLEMENT
    assert transition("implement", TaskEvent.REQUIREMENTS_SATISFIED).after == TaskStage.IMPLEMENT
    assert transition("implement", TaskEvent.VERIFICATION_REQUESTED).after == TaskStage.VERIFY
    assert transition("verify", TaskEvent.VERIFICATION_FAILED).after == TaskStage.REPAIR
    assert transition("repair", TaskEvent.MUTATION_SUCCEEDED).after == TaskStage.VERIFY
    assert transition("verify", TaskEvent.REQUIREMENTS_SATISFIED).after == TaskStage.COMPLETE


def test_tool_events_are_grounded_in_success() -> None:
    assert event_for_tool("write_file", succeeded=True) == TaskEvent.MUTATION_SUCCEEDED
    assert event_for_tool("write_file", succeeded=False) is None
    assert event_for_tool("bash", succeeded=False, verification=True) == TaskEvent.VERIFICATION_FAILED
