"""Regression: an Esc-interrupted (cancelled) agent turn must finalize.

_interrupt_turn calls `worker.cancel()`, so the run_agent_turn worker lands in
WorkerState.CANCELLED — not SUCCESS. Before the fix, on_worker_state_changed
handled only SUCCESS/ERROR, so a cancel left the thinking indicator animating
and `_agent_busy` stuck True (input wedged). The CANCELLED branch must call
_on_turn_done() for the agent-turn worker (and only that worker).
"""
from __future__ import annotations

import types

from textual.worker import WorkerState

from localcode.tui.screens.chat import ChatScreen


def _event(state, name):
    return types.SimpleNamespace(
        state=state,
        worker=types.SimpleNamespace(name=name, error=None),
    )


def test_cancelled_agent_turn_calls_on_turn_done():
    calls = {"done": 0}
    fake = types.SimpleNamespace(_on_turn_done=lambda: calls.__setitem__("done", calls["done"] + 1))
    ChatScreen.on_worker_state_changed(fake, _event(WorkerState.CANCELLED, "run_agent_turn"))
    assert calls["done"] == 1


def test_cancelled_non_agent_worker_does_not_finalize_turn():
    calls = {"done": 0}
    fake = types.SimpleNamespace(_on_turn_done=lambda: calls.__setitem__("done", calls["done"] + 1))
    # a cancelled helper worker (e.g. a model-swap probe) must not end the turn
    ChatScreen.on_worker_state_changed(fake, _event(WorkerState.CANCELLED, "some_helper"))
    assert calls["done"] == 0
