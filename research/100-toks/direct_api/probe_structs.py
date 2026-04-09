#!/usr/bin/env python3
"""Probe actual struct layout from the compiled llama.cpp library."""
import ctypes
import struct

LLAMA_LIB = "/Users/marcsolomon/llama-cpp-turboquant/build/bin/libllama.dylib"
GGML_LIB = "/Users/marcsolomon/llama-cpp-turboquant/build/bin/libggml.dylib"

ggml = ctypes.CDLL(GGML_LIB)
lib = ctypes.CDLL(LLAMA_LIB)

# Define the struct as we think it should be
class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx",              ctypes.c_uint32),
        ("n_batch",            ctypes.c_uint32),
        ("n_ubatch",           ctypes.c_uint32),
        ("n_seq_max",          ctypes.c_uint32),
        ("n_threads",          ctypes.c_int32),
        ("n_threads_batch",    ctypes.c_int32),
        ("rope_scaling_type",  ctypes.c_int32),
        ("pooling_type",       ctypes.c_int32),
        ("attention_type",     ctypes.c_int32),
        ("flash_attn_type",    ctypes.c_int32),
        ("rope_freq_base",     ctypes.c_float),
        ("rope_freq_scale",    ctypes.c_float),
        ("yarn_ext_factor",    ctypes.c_float),
        ("yarn_attn_factor",   ctypes.c_float),
        ("yarn_beta_fast",     ctypes.c_float),
        ("yarn_beta_slow",     ctypes.c_float),
        ("yarn_orig_ctx",      ctypes.c_uint32),
        ("defrag_thold",       ctypes.c_float),
        ("cb_eval",            ctypes.c_void_p),
        ("cb_eval_user_data",  ctypes.c_void_p),
        ("type_k",             ctypes.c_int32),
        ("type_v",             ctypes.c_int32),
        ("abort_callback",     ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings",         ctypes.c_bool),
        ("offload_kqv",        ctypes.c_bool),
        ("no_perf",            ctypes.c_bool),
        ("op_offload",         ctypes.c_bool),
        ("swa_full",           ctypes.c_bool),
        ("kv_unified",         ctypes.c_bool),
        ("samplers",           ctypes.c_void_p),
        ("n_samplers",         ctypes.c_size_t),
    ]

lib.llama_context_default_params.restype = llama_context_params
lib.llama_context_default_params.argtypes = []

print(f"Python sizeof(llama_context_params) = {ctypes.sizeof(llama_context_params)}")
print()

params = lib.llama_context_default_params()

# Print all field values
for name, _ in llama_context_params._fields_:
    val = getattr(params, name)
    field_desc = getattr(llama_context_params, name)
    print(f"  {name:30s} offset={field_desc.offset:3d}  value={val}")

# Get raw bytes
raw = bytes(ctypes.string_at(ctypes.addressof(params), ctypes.sizeof(params)))
print(f"\nRaw hex dump ({len(raw)} bytes):")
for i in range(0, len(raw), 8):
    chunk = raw[i:i+8]
    hex_str = " ".join(f"{b:02x}" for b in chunk)
    print(f"  {i:3d}: {hex_str:23s}")

# Verify type_k at offset 88
print(f"\ntype_k at offset 88 = {struct.unpack_from('i', raw, 88)[0]}")
print(f"type_v at offset 92 = {struct.unpack_from('i', raw, 92)[0]}")

# Test: set type_k and type_v and verify
params.type_k = 8  # Q8_0
params.type_v = 42  # TURBO4
raw2 = bytes(ctypes.string_at(ctypes.addressof(params), ctypes.sizeof(params)))
print(f"\nAfter setting type_k=8, type_v=42:")
print(f"type_k at offset 88 = {struct.unpack_from('i', raw2, 88)[0]}")
print(f"type_v at offset 92 = {struct.unpack_from('i', raw2, 92)[0]}")
print(f"Bytes [88:96] = {' '.join(f'{b:02x}' for b in raw2[88:96])}")

print("\n=== llama_model_params ===")

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices",                   ctypes.c_void_p),
        ("tensor_buft_overrides",     ctypes.c_void_p),
        ("n_gpu_layers",              ctypes.c_int32),
        ("split_mode",                ctypes.c_int32),
        ("main_gpu",                  ctypes.c_int32),
        ("tensor_split",              ctypes.c_void_p),
        ("progress_callback",         ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides",              ctypes.c_void_p),
        ("vocab_only",                ctypes.c_bool),
        ("use_mmap",                  ctypes.c_bool),
        ("use_direct_io",             ctypes.c_bool),
        ("use_mlock",                 ctypes.c_bool),
        ("check_tensors",             ctypes.c_bool),
        ("use_extra_bufts",           ctypes.c_bool),
        ("no_host",                   ctypes.c_bool),
        ("no_alloc",                  ctypes.c_bool),
    ]

lib.llama_model_default_params.restype = llama_model_params
lib.llama_model_default_params.argtypes = []

mparams = lib.llama_model_default_params()
print(f"sizeof = {ctypes.sizeof(llama_model_params)}")
for name, _ in llama_model_params._fields_:
    val = getattr(mparams, name)
    field_desc = getattr(llama_model_params, name)
    print(f"  {name:30s} offset={field_desc.offset:3d}  value={val}")
