from localcode.execution_policy import assess_shell_execution
from localcode.thinking import adaptive_reasoning_policy
from localcode.agent.context import latency_budgeted_hot_replay
from localcode.agent.hooks import CompletionEvidence, deterministic_completion


def test_shell_exit_zero_is_not_task_success_when_failure_is_masked():
    result = assess_shell_execution("pytest -q || true", "2 tests failed", 0)
    assert result.shell_succeeded
    assert not result.task_succeeded
    assert "failure-masking-control-flow" in result.signals


def test_pipeline_without_pipefail_lowers_confidence():
    result = assess_shell_execution("build | tail -20", "build complete", 0)
    assert result.task_succeeded
    assert result.confidence == "low"


def test_adaptive_reasoning_spends_tokens_on_surprises_not_polling():
    assert adaptive_reasoning_policy("debugging")
    assert adaptive_reasoning_policy("build", unexpected_failure=True)
    assert not adaptive_reasoning_policy("poll")
    assert not adaptive_reasoning_policy("write")


def test_slow_ttft_reduces_hot_replay_but_not_below_floor():
    normal = latency_budgeted_hot_replay(900_000, 4_000)
    slow = latency_budgeted_hot_replay(900_000, 32_000)
    assert slow[0] < normal[0]
    assert slow[0] >= 18_000
    assert slow[1] >= 2


def test_deterministic_completion_requires_evidence_and_stops_when_done():
    missing = deterministic_completion([CompletionEvidence("tests pass")])
    assert not missing.ok
    done = deterministic_completion([
        CompletionEvidence("tests pass", ("pytest: 12 passed",), True),
        CompletionEvidence("feature exists", ("src/app.py:42",), True),
    ])
    assert done.ok
    assert "Stop" in done.correction
