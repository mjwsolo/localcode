from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf

from .qwen import Qwen3MoeModel


@ModelBase.register("Kolibri1ForCausalLM")
@ModelBase.example("Aleph-Alpha/Kolibri-1")
class Kolibri1Model(Qwen3MoeModel):
    """Aleph Alpha Kolibri 1: Qwen3-MoE with sandwich norms, one ungated shared
    expert, interleaved sliding-window (RoPE) / full (NoPE) attention, and a
    router that selects top-k on logits + expert_bias but weights by the
    unbiased sigmoid(logits)."""

    model_arch = gguf.MODEL_ARCH.KOLIBRI1

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        layer_types = self.hparams["layer_types"]
        if len(layer_types) != self.block_count:
            raise ValueError(f"layer_types has {len(layer_types)} entries, expected {self.block_count}")
        unknown = set(layer_types) - {"sliding_attention", "full_attention"}
        if unknown:
            raise ValueError(f"Unsupported Kolibri 1 layer_types: {sorted(unknown)}")

        sliding_window = self.hparams.get("sliding_window")
        if not sliding_window or sliding_window <= 0:
            raise ValueError(f"Kolibri 1 needs a positive sliding_window, got {sliding_window}")

        self.gguf_writer.add_sliding_window(sliding_window)
        self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in layer_types])

        self.gguf_writer.add_expert_shared_count(1)
        self.gguf_writer.add_expert_weights_norm(bool(self.hparams.get("norm_topk_prob", False)))
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID_LOGIT_ADD)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # model.layers.N.moe.router.expert_bias -> exp_probs_b.bias
        if name.endswith(".moe.router.expert_bias"):
            name = name + ".bias"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the shared expert would otherwise be swallowed by the routed-expert merge in Qwen2MoeModel
        if ".mlp.shared_experts." in name:
            yield from ModelBase.modify_tensors(self, data_torch, name, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)
