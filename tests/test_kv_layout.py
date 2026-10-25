"""KV sizing reads the per-layer layout: Gemma 4 sliding windows, Qwen 3.5+ hybrids."""
from localcode.runtime import LocalCodeRuntimeGateway, RuntimeConfig, _kv_layers


def gateway_with(meta):
    g = LocalCodeRuntimeGateway(RuntimeConfig())
    g._gguf_kv_meta = lambda *_a, **_k: meta
    g._effective_kv_dtypes = lambda *_a, **_k: ("q8_0", "q8_0")
    return g


def test_gemma4_only_full_attention_layers_grow_with_context():
    # Gemma 4 12B: 48 layers, 5 sliding (8 heads x 256) then 1 global (1 head x 512).
    found = {"kv_heads_list": [8, 8, 8, 8, 8, 1], "swa_pattern": [True] * 5 + [False],
             "sliding_window": 1024, "head_dim_swa": 256, "value_head_dim_swa": 256}
    layers = _kv_layers(48, 8, 512, 512, found)
    assert len(layers) == 48
    assert sum(1 for *_x, sliding in layers if not sliding) == 8
    meta = {"n_layers": 48, "n_kv_heads": 8, "head_dim": 512, "value_head_dim": 512, "layers": layers, "sliding_window": 1024}
    g = gateway_with(meta)
    q8 = g._KV_DTYPE_BYTES["q8_0"]
    assert g._kv_bytes_per_token("m.gguf") == 8 * 1 * (512 * q8 + 512 * q8)
    # Sliding layers are a fixed block: window plus a batch, with a copy per checkpoint.
    fixed = g._kv_fixed_bytes("m.gguf")
    assert fixed == int(40 * 8 * (256 * q8 + 256 * q8) * (1024 + 2048) * 5)
    # Far below the old estimate: all 48 layers as full attention with 16 heads.
    assert g._kv_bytes_per_token("m.gguf") < 48 * 16 * 1024 * q8 / 50


def test_qwen_hybrid_keeps_kv_on_every_fourth_layer_and_counts_its_state():
    found = {"full_attention_interval": 4}
    layers = _kv_layers(40, 2, 256, 256, found)
    assert len(layers) == 10 and all(not sliding for *_x, sliding in layers)
    meta = {"n_layers": 40, "n_kv_heads": 2, "head_dim": 256, "value_head_dim": 256, "layers": layers,
            "recurrent_state_bytes": 30 * 4096 * (128 + 4) * 4}
    g = gateway_with(meta)
    q8 = g._KV_DTYPE_BYTES["q8_0"]
    assert g._kv_bytes_per_token("m.gguf") == 10 * 2 * (256 * q8 + 256 * q8)
    assert g._kv_fixed_bytes("m.gguf") == 30 * 4096 * 132 * 4 * 5


def test_plain_models_are_unchanged():
    layers = _kv_layers(32, 8, 128, 128, {})
    assert len(layers) == 32
    meta = {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128, "value_head_dim": 128}
    g = gateway_with(meta)
    q8 = g._KV_DTYPE_BYTES["q8_0"]
    assert g._kv_bytes_per_token("m.gguf") == 8 * (32 * 128 * q8 + 32 * 128 * q8)
    assert g._kv_fixed_bytes("m.gguf") == 0
