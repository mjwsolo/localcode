"""Provider/model-specific reasoning protocol capabilities.

Reasoning is not one portable boolean: local chat templates expose different
controls and some cannot enforce a separate budget.  Keep those differences in
one typed registry so request construction and recovery use the same facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["ReasoningControl", "ReasoningCapabilities", "reasoning_capabilities"]


class ReasoningControl(str, Enum):
    CHAT_TEMPLATE = "chat_template"
    EFFORT = "reasoning_effort"
    NONE = "none"


@dataclass(frozen=True)
class ReasoningCapabilities:
    family: str
    supported: bool
    control: ReasoningControl
    supports_budget: bool
    preserves_reasoning: bool = True
    supports_parallel_tools: bool = False


def reasoning_capabilities(model: str, provider: str = "llama_cpp") -> ReasoningCapabilities:
    """Resolve the effective reasoning protocol from stable model/provider hints."""
    name = (model or "").lower()
    provider = (provider or "").lower()
    if "diffusion" in name:
        return ReasoningCapabilities("diffusion", False, ReasoningControl.NONE, False, False)
    if provider in {"openai", "responses"}:
        return ReasoningCapabilities("openai", True, ReasoningControl.EFFORT, False, True, True)
    try:
        from .models_catalog import by_filename, by_key
        choice = by_filename(model) or by_key(model)
    except Exception:
        choice = None
    if choice is not None:
        control = ReasoningControl(choice.reasoning_control)
        return ReasoningCapabilities(
            str(choice.architecture),
            control is not ReasoningControl.NONE,
            control,
            choice.reasoning_budget_tokens > 0,
            choice.preserves_reasoning,
            choice.supports_parallel_tools,
        )
    family = next((f for f in ("qwen", "deepseek", "gemma", "cohere") if f in name), "generic")
    # llama.cpp exposes enable_thinking through chat-template kwargs. Its
    # separate token budget only works when the selected template declares a
    # recognizable reasoning terminator; the known reasoning families do.
    budget = family in {"qwen", "deepseek", "gemma"}
    return ReasoningCapabilities(family, True, ReasoningControl.CHAT_TEMPLATE, budget)
