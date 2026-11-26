from gguf_builder import moe_tensors, write_gguf

from localcode.gguf_experts import read_moe_info

KV = {"qwen3moe.expert_count": 8, "qwen3moe.expert_used_count": 2, "qwen3moe.block_count": 3}


def test_sums_only_expert_tensors(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", moe_tensors(3, 8192), KV)
    info = read_moe_info(p)
    assert info.expert_bytes == 3 * 3 * 8192
    assert (info.n_expert, info.n_expert_used, info.n_moe_layers) == (8, 2, 3)
    assert info.per_expert_bytes == 3 * 8192 // 8
    assert info.min_bank_bytes == 2 * 3 * info.per_expert_bytes


def test_dense_model_has_no_expert_info(tmp_path):
    p = write_gguf(tmp_path / "d.gguf", [("token_embd.weight", 4096), ("blk.0.ffn_down.weight", 4096)], {})
    assert read_moe_info(p) is None


def test_split_gguf_adds_every_shard(tmp_path):
    a = write_gguf(tmp_path / "m-00001-of-00002.gguf", moe_tensors(2, 8192), KV)
    write_gguf(tmp_path / "m-00002-of-00002.gguf", [
        (f"blk.{i}.ffn_down_exps.weight", 8192) for i in (2, 3)], {})
    info = read_moe_info(a)
    assert info.expert_bytes == 2 * 3 * 8192 + 2 * 8192
    assert info.n_moe_layers == 4


def test_unreadable_file_is_none(tmp_path):
    (tmp_path / "x.gguf").write_bytes(b"nope")
    assert read_moe_info(tmp_path / "x.gguf") is None
    assert read_moe_info(tmp_path / "missing.gguf") is None
