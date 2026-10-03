from localcode.reasoning_capabilities import ReasoningControl, reasoning_capabilities

def test_local_reasoning_family_matrix():
    for model in ("qwen3.gguf", "deepseek-r1.gguf", "gemma-4.gguf"):
        caps = reasoning_capabilities(model)
        assert caps.supported
        assert caps.control is ReasoningControl.CHAT_TEMPLATE
        assert caps.supports_budget


def test_non_reasoning_and_hosted_protocols_are_explicit():
    assert not reasoning_capabilities("diffusion-gemma.gguf").supported
    hosted = reasoning_capabilities("gpt-5", "openai")
    assert hosted.control is ReasoningControl.EFFORT
    assert hosted.supports_parallel_tools


def test_catalog_is_authoritative_for_reasoning_protocol():
    diffusion = reasoning_capabilities("diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
    assert diffusion.control is ReasoningControl.NONE
    cohere = reasoning_capabilities("North-Mini-Code-1.0-UD-Q4_K_M.gguf")
    assert cohere.supported and not cohere.supports_budget
