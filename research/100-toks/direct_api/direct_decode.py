#!/usr/bin/env python3
"""
Direct llama.cpp C API bridge via ctypes — bypass HTTP server entirely.

Measures per-token decode latency at each step:
  1. llama_decode() call
  2. llama_synchronize()
  3. llama_get_logits() + argmax
  4. Total per-token time

Compares against HTTP server baseline (if running on :8081).

Usage:
  python direct_decode.py [--tokens N] [--no-http]
"""

import ctypes
import ctypes.util
import time
import sys
import os
import argparse
import struct as pystruct
import subprocess

# ─── Paths ───────────────────────────────────────────────────────────────────

LLAMA_LIB = "/Users/marcsolomon/llama-cpp-turboquant/build/bin/libllama.dylib"
GGML_LIB  = "/Users/marcsolomon/llama-cpp-turboquant/build/bin/libggml.dylib"
MODEL_PATH = "/Users/marcsolomon/.ollama/models/blobs/sha256-dc8031f1f9de58ff7e460632ab0190b4f42ff95e64669a3ea93afe09b83e0e56"

# ─── GGML type constants ─────────────────────────────────────────────────────

GGML_TYPE_F32     = 0
GGML_TYPE_Q8_0    = 8
GGML_TYPE_TURBO4  = 42  # turbo4 KV cache

# ─── llama enums ──────────────────────────────────────────────────────────────

LLAMA_FLASH_ATTN_ENABLED = 1

# ─── Struct definitions ──────────────────────────────────────────────────────
#
# CRITICAL: These must match the C struct layouts exactly. On ARM64 macOS,
# structs are passed by value with natural alignment. We use the library's
# own default_params functions to get correct initial values and only
# patch the fields we need.

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens",  ctypes.c_int32),
        # ctypes automatically adds 4 bytes padding here for pointer alignment
        ("token",     ctypes.POINTER(ctypes.c_int32)),
        ("embd",      ctypes.c_void_p),
        ("pos",       ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id",  ctypes.POINTER(ctypes.c_int32)),
        ("seq_id",    ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits",    ctypes.POINTER(ctypes.c_int8)),
    ]

    def __repr__(self):
        return (f"llama_batch(n_tokens={self.n_tokens}, "
                f"token={self.token}, pos={self.pos}, "
                f"logits={self.logits})")


# Instead of defining the exact struct layouts (which are fragile),
# we'll use opaque byte buffers and patch fields at known offsets.
# The offsets are determined from the header:
#
# llama_model_params layout (ARM64, natural alignment):
#   0:  devices              (ptr, 8 bytes)
#   8:  tensor_buft_overrides (ptr, 8 bytes)
#   16: n_gpu_layers         (i32, 4 bytes)
#   20: split_mode           (i32/enum, 4 bytes)
#   24: main_gpu             (i32, 4 bytes)
#   28: pad                  (4 bytes for pointer alignment)
#   32: tensor_split         (ptr, 8 bytes)
#   40: progress_callback    (ptr, 8 bytes)
#   48: progress_callback_user_data (ptr, 8 bytes)
#   56: kv_overrides         (ptr, 8 bytes)
#   64: vocab_only           (bool, 1 byte)
#   65: use_mmap             (bool, 1 byte)
#   66: use_direct_io        (bool, 1 byte)
#   67: use_mlock            (bool, 1 byte)
#   68: check_tensors        (bool, 1 byte)
#   69: use_extra_bufts      (bool, 1 byte)
#   70: no_host              (bool, 1 byte)
#   71: no_alloc             (bool, 1 byte)
#   Total: 72 bytes (8-byte aligned)

MODEL_PARAMS_SIZE = 72  # We'll verify this

# llama_context_params is much larger and more complex.
# We'll define it as a proper ctypes struct.
class llama_context_params(ctypes.Structure):
    # Layout verified from llama.h struct definition.
    # On ARM64 macOS: pointers 8 bytes, int32 4 bytes, float 4 bytes, bool 1 byte.
    # ctypes handles natural alignment automatically (no manual padding needed
    # EXCEPT where int32 precedes a pointer and ctypes wouldn't know to pad).
    # Actually, ctypes DOES handle this correctly with natural alignment.
    _fields_ = [
        ("n_ctx",              ctypes.c_uint32),    # 0
        ("n_batch",            ctypes.c_uint32),    # 4
        ("n_ubatch",           ctypes.c_uint32),    # 8
        ("n_seq_max",          ctypes.c_uint32),    # 12
        ("n_threads",          ctypes.c_int32),     # 16
        ("n_threads_batch",    ctypes.c_int32),     # 20
        ("rope_scaling_type",  ctypes.c_int32),     # 24
        ("pooling_type",       ctypes.c_int32),     # 28
        ("attention_type",     ctypes.c_int32),     # 32
        ("flash_attn_type",    ctypes.c_int32),     # 36
        ("rope_freq_base",     ctypes.c_float),     # 40
        ("rope_freq_scale",    ctypes.c_float),     # 44
        ("yarn_ext_factor",    ctypes.c_float),     # 48
        ("yarn_attn_factor",   ctypes.c_float),     # 52
        ("yarn_beta_fast",     ctypes.c_float),     # 56
        ("yarn_beta_slow",     ctypes.c_float),     # 60
        ("yarn_orig_ctx",      ctypes.c_uint32),    # 64
        ("defrag_thold",       ctypes.c_float),     # 68
        # 72: ctypes auto-pads 4 bytes here before next pointer
        ("cb_eval",            ctypes.c_void_p),    # 72 -> aligned to 80 by ctypes? NO:
        # Actually at offset 72, which is 8-byte aligned (72/8=9), so ptr goes here
        ("cb_eval_user_data",  ctypes.c_void_p),    # 80
        ("type_k",             ctypes.c_int32),     # 88
        ("type_v",             ctypes.c_int32),     # 92
        # 96: ctypes auto-pads before next pointer? 96 is 8-byte aligned, so no pad
        ("abort_callback",     ctypes.c_void_p),    # 96
        ("abort_callback_data", ctypes.c_void_p),   # 104
        # booleans at offset 112
        ("embeddings",         ctypes.c_bool),      # 112
        ("offload_kqv",        ctypes.c_bool),      # 113
        ("no_perf",            ctypes.c_bool),      # 114
        ("op_offload",         ctypes.c_bool),      # 115
        ("swa_full",           ctypes.c_bool),      # 116
        ("kv_unified",         ctypes.c_bool),      # 117
        # 118: ctypes auto-pads to 8-byte alignment for next pointer (6 bytes pad)
        ("samplers",           ctypes.c_void_p),    # 120 (after 2 bytes pad)
        ("n_samplers",         ctypes.c_size_t),    # 128
    ]


class llama_perf_context_data(ctypes.Structure):
    _fields_ = [
        ("t_start_ms",   ctypes.c_double),
        ("t_load_ms",    ctypes.c_double),
        ("t_p_eval_ms",  ctypes.c_double),
        ("t_eval_ms",    ctypes.c_double),
        ("n_p_eval",     ctypes.c_int32),
        ("n_eval",       ctypes.c_int32),
        ("n_reused",     ctypes.c_int32),
    ]


# ─── Opaque model params approach ────────────────────────────────────────────
# Since struct layout is fragile, we use a raw byte buffer approach for
# llama_model_params. We call the default_params function to get valid
# defaults, then patch specific fields at their offsets.

class OpaqueModelParams:
    """Wraps llama_model_params as raw bytes, patching fields by offset."""

    # Offsets (ARM64 macOS, verified from header field order + alignment):
    OFF_N_GPU_LAYERS = 16
    OFF_USE_MMAP = 65

    def __init__(self, raw_bytes):
        self._buf = bytearray(raw_bytes)

    def set_n_gpu_layers(self, val):
        pystruct.pack_into('i', self._buf, self.OFF_N_GPU_LAYERS, val)

    def set_use_mmap(self, val):
        self._buf[self.OFF_USE_MMAP] = 1 if val else 0

    def as_bytes(self):
        return bytes(self._buf)


# ─── Load libraries ──────────────────────────────────────────────────────────

def load_libs():
    """Load libggml and libllama, return the llama cdll."""
    ggml = ctypes.CDLL(GGML_LIB)
    llama = ctypes.CDLL(LLAMA_LIB)
    return llama, ggml


def setup_bindings(lib):
    """Set up ctypes function signatures for the llama C API."""

    # Backend init
    lib.llama_backend_init.argtypes = []
    lib.llama_backend_init.restype = None

    lib.llama_backend_free.argtypes = []
    lib.llama_backend_free.restype = None

    # Default params — return raw bytes, we'll interpret them manually
    # Actually, for model_params we need a different approach.
    # Let's define the struct properly but verify size.

    # We'll use a helper to get default params as raw bytes
    # by calling the function with a large enough return buffer.

    # Model load — takes path + struct by value
    # We'll handle this with a wrapper function below

    # Context default params
    lib.llama_context_default_params.argtypes = []
    lib.llama_context_default_params.restype = llama_context_params

    # Context creation
    lib.llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
    lib.llama_init_from_model.restype = ctypes.c_void_p

    lib.llama_free.argtypes = [ctypes.c_void_p]
    lib.llama_free.restype = None

    # Model free
    lib.llama_model_free.argtypes = [ctypes.c_void_p]
    lib.llama_model_free.restype = None

    # Vocab
    lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    lib.llama_model_get_vocab.restype = ctypes.c_void_p

    lib.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_n_tokens.restype = ctypes.c_int32

    lib.llama_vocab_bos.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_bos.restype = ctypes.c_int32

    lib.llama_vocab_eos.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_eos.restype = ctypes.c_int32

    # Tokenize
    lib.llama_tokenize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ctypes.c_bool, ctypes.c_bool,
    ]
    lib.llama_tokenize.restype = ctypes.c_int32

    # Token to piece
    lib.llama_token_to_piece.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_bool,
    ]
    lib.llama_token_to_piece.restype = ctypes.c_int32

    # Batch
    lib.llama_batch_get_one.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    lib.llama_batch_get_one.restype = llama_batch

    lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    lib.llama_batch_init.restype = llama_batch

    lib.llama_batch_free.argtypes = [llama_batch]
    lib.llama_batch_free.restype = None

    # Decode
    lib.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
    lib.llama_decode.restype = ctypes.c_int32

    # Synchronize
    lib.llama_synchronize.argtypes = [ctypes.c_void_p]
    lib.llama_synchronize.restype = None

    # Logits
    lib.llama_get_logits.argtypes = [ctypes.c_void_p]
    lib.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    lib.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

    # Perf
    lib.llama_perf_context.argtypes = [ctypes.c_void_p]
    lib.llama_perf_context.restype = llama_perf_context_data

    lib.llama_perf_context_reset.argtypes = [ctypes.c_void_p]
    lib.llama_perf_context_reset.restype = None

    lib.llama_perf_context_print.argtypes = [ctypes.c_void_p]
    lib.llama_perf_context_print.restype = None

    # n_ctx
    lib.llama_n_ctx.argtypes = [ctypes.c_void_p]
    lib.llama_n_ctx.restype = ctypes.c_uint32

    return lib


# ─── Safe model loading ──────────────────────────────────────────────────────
# The problem with passing structs by value through ctypes is alignment.
# On ARM64, the ABI passes small structs in registers and larger ones
# on the stack. If our Python struct doesn't match, we get corruption.
#
# SOLUTION: Use llama_load_model_from_file (the deprecated name) which
# has the same ABI, OR define the model params struct correctly.
#
# Actually, let's just use ctypes.CFUNCTYPE to define the exact calling
# convention and pass the struct as raw bytes.

def load_model_safe(lib, model_path, n_gpu_layers=999, use_mmap=True):
    """Load model using a safe approach that handles struct alignment correctly."""

    # Define model_params struct that matches C exactly.
    # On ARM64 macOS: pointers are 8 bytes, ints are 4 bytes, bools are 1 byte.
    # Struct is naturally aligned.
    class llama_model_params(ctypes.Structure):
        _fields_ = [
            ("devices",                   ctypes.c_void_p),     # 0
            ("tensor_buft_overrides",     ctypes.c_void_p),     # 8
            ("n_gpu_layers",              ctypes.c_int32),      # 16
            ("split_mode",                ctypes.c_int32),      # 20
            ("main_gpu",                  ctypes.c_int32),      # 24
            ("_pad0",                     ctypes.c_int32),      # 28 (alignment padding for tensor_split ptr)
            ("tensor_split",              ctypes.c_void_p),     # 32
            ("progress_callback",         ctypes.c_void_p),     # 40
            ("progress_callback_user_data", ctypes.c_void_p),   # 48
            ("kv_overrides",              ctypes.c_void_p),     # 56
            ("vocab_only",                ctypes.c_bool),       # 64
            ("use_mmap",                  ctypes.c_bool),       # 65
            ("use_direct_io",             ctypes.c_bool),       # 66
            ("use_mlock",                 ctypes.c_bool),       # 67
            ("check_tensors",             ctypes.c_bool),       # 68
            ("use_extra_bufts",           ctypes.c_bool),       # 69
            ("no_host",                   ctypes.c_bool),       # 70
            ("no_alloc",                  ctypes.c_bool),       # 71
        ]
        _pack_ = 1  # Don't let ctypes add extra padding

    # Actually, _pack_ = 1 is wrong for this struct. The C compiler uses
    # natural alignment. Let me NOT use _pack_ and instead let ctypes
    # figure out the alignment naturally. But the main_gpu -> tensor_split
    # gap needs explicit padding since ctypes knows int32 doesn't need
    # 8-byte alignment.

    # Let's verify: without _pack_, ctypes will:
    # - c_void_p at offset 0 (8 bytes)
    # - c_void_p at offset 8 (8 bytes)
    # - c_int32 at offset 16 (4 bytes)
    # - c_int32 at offset 20 (4 bytes) [split_mode]
    # - c_int32 at offset 24 (4 bytes) [main_gpu]
    # - c_int32 _pad at offset 28 (4 bytes)
    # - c_void_p at offset 32 (8 bytes) [tensor_split]
    # This should be correct.

    # But wait - without _pack_, ctypes uses the alignment of the largest
    # field type. On 64-bit, it would add the padding automatically between
    # main_gpu (ends at 28) and tensor_split (needs 8-byte alignment at 32).
    # So we DON'T need explicit _pad0.

    class llama_model_params_v2(ctypes.Structure):
        _fields_ = [
            ("devices",                   ctypes.c_void_p),
            ("tensor_buft_overrides",     ctypes.c_void_p),
            ("n_gpu_layers",              ctypes.c_int32),
            ("split_mode",                ctypes.c_int32),
            ("main_gpu",                  ctypes.c_int32),
            # ctypes will add 4 bytes padding here automatically
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

    # Set up the function with proper types
    lib.llama_model_default_params.argtypes = []
    lib.llama_model_default_params.restype = llama_model_params_v2

    lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params_v2]
    lib.llama_model_load_from_file.restype = ctypes.c_void_p

    # Get defaults
    mparams = lib.llama_model_default_params()

    # Print struct info for debugging
    print(f"    sizeof(model_params) = {ctypes.sizeof(mparams)} (Python)")
    print(f"    default n_gpu_layers = {mparams.n_gpu_layers}")
    print(f"    default use_mmap = {mparams.use_mmap}")
    print(f"    default split_mode = {mparams.split_mode}")

    # Override fields
    mparams.n_gpu_layers = n_gpu_layers
    mparams.use_mmap = use_mmap

    print(f"    set n_gpu_layers = {mparams.n_gpu_layers}")
    print(f"    set use_mmap = {mparams.use_mmap}")

    # Load
    model = lib.llama_model_load_from_file(model_path.encode("utf-8"), mparams)
    return model


# ─── Token helpers ────────────────────────────────────────────────────────────

def token_to_str(lib, vocab, token_id):
    buf = ctypes.create_string_buffer(256)
    n = lib.llama_token_to_piece(vocab, token_id, buf, 256, 0, False)
    if n < 0:
        return f"<tok:{token_id}>"
    return buf.value[:n].decode("utf-8", errors="replace")


def tokenize(lib, vocab, text, add_special=True):
    text_bytes = text.encode("utf-8")
    n = lib.llama_tokenize(vocab, text_bytes, len(text_bytes), None, 0, add_special, True)
    n_tokens = abs(n)
    tokens = (ctypes.c_int32 * n_tokens)()
    n2 = lib.llama_tokenize(vocab, text_bytes, len(text_bytes), tokens, n_tokens, add_special, True)
    if n2 < 0:
        raise RuntimeError(f"tokenize failed: {n2}")
    return tokens, n2


# ─── argmax via ctypes (no numpy dependency) ──────────────────────────────────

def argmax_logits_fast(logits_ptr, n_vocab):
    """Read logits from C pointer, return argmax token id using ctypes array."""
    # Cast to array for fast access
    arr = ctypes.cast(logits_ptr, ctypes.POINTER(ctypes.c_float * n_vocab)).contents
    best_id = 0
    best_val = arr[0]
    for i in range(1, n_vocab):
        if arr[i] > best_val:
            best_val = arr[i]
            best_id = i
    return best_id


def argmax_logits(logits_ptr, n_vocab):
    """Argmax using numpy for speed."""
    try:
        import numpy as np
        arr = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))
        return int(np.argmax(arr))
    except ImportError:
        return argmax_logits_fast(logits_ptr, n_vocab)


# ─── DIRECT API DECODE ───────────────────────────────────────────────────────

def run_direct_decode(n_gen_tokens=20, use_turboquant=True):
    """Load model, tokenize prompt, decode tokens one at a time, measure everything."""

    # Force unbuffered output so we can see progress before crash
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    print("=" * 70, flush=True)
    print("DIRECT llama.cpp C API DECODE BENCHMARK", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    # Load libraries
    print("[1/5] Loading shared libraries...", flush=True)
    lib, _ = load_libs()
    print("    Libraries loaded OK", flush=True)
    lib = setup_bindings(lib)
    print("    Bindings set up OK", flush=True)

    # Init backend
    print("[2/5] Initializing backend...", flush=True)
    lib.llama_backend_init()
    print("    Backend initialized OK", flush=True)

    # Load model
    print("[3/5] Loading model (this takes ~10-30s)...", flush=True)
    t_load_start = time.perf_counter()
    model = load_model_safe(lib, MODEL_PATH, n_gpu_layers=999, use_mmap=True)
    t_load = time.perf_counter() - t_load_start
    if not model:
        print("FATAL: Failed to load model!")
        print("  Retrying with n_gpu_layers=0 (CPU only)...")
        t_load_start = time.perf_counter()
        model = load_model_safe(lib, MODEL_PATH, n_gpu_layers=0, use_mmap=True)
        t_load = time.perf_counter() - t_load_start
        if not model:
            print("FATAL: Model load failed even with CPU-only!")
            sys.exit(1)
    print(f"    Model loaded in {t_load:.1f}s")

    # Get vocab info
    vocab = lib.llama_model_get_vocab(model)
    n_vocab = lib.llama_vocab_n_tokens(vocab)
    eos_token = lib.llama_vocab_eos(vocab)
    bos_token = lib.llama_vocab_bos(vocab)
    print(f"    Vocab size: {n_vocab}, BOS: {bos_token}, EOS: {eos_token}")

    # Create context
    print("[4/5] Creating context...")
    cparams = lib.llama_context_default_params()

    # Print context params struct info for debugging
    print(f"    sizeof(context_params) = {ctypes.sizeof(cparams)} (Python)")
    print(f"    default n_ctx = {cparams.n_ctx}")
    print(f"    default n_batch = {cparams.n_batch}")
    print(f"    default type_k = {cparams.type_k}")
    print(f"    default type_v = {cparams.type_v}")

    cparams.n_ctx = 4096
    cparams.n_batch = 2048
    cparams.n_ubatch = 512
    cparams.n_threads = 10
    cparams.n_threads_batch = 10
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_ENABLED
    cparams.offload_kqv = True
    cparams.op_offload = True

    if use_turboquant:
        cparams.type_k = GGML_TYPE_Q8_0
        cparams.type_v = GGML_TYPE_TURBO4
        print("    KV cache: q8_0-K + turbo4-V (TurboQuant)")
    else:
        cparams.type_k = GGML_TYPE_Q8_0
        cparams.type_v = GGML_TYPE_Q8_0
        print("    KV cache: q8_0 (standard)")

    ctx = lib.llama_init_from_model(model, cparams)
    if not ctx:
        print("FATAL: Failed to create context!")
        print("  This usually means struct alignment mismatch.")
        print(f"  sizeof(llama_context_params) = {ctypes.sizeof(cparams)}")
        lib.llama_model_free(model)
        sys.exit(1)
    actual_ctx = lib.llama_n_ctx(ctx)
    print(f"    Context created (n_ctx={actual_ctx})")

    # Tokenize prompt
    prompt = "<start_of_turn>user\nWrite a Python function to compute fibonacci numbers efficiently.<end_of_turn>\n<start_of_turn>model\n"
    tokens, n_tokens = tokenize(lib, vocab, prompt, add_special=True)
    print(f"\n[5/5] Prompt: {n_tokens} tokens")

    # ─── Phase 1: Prompt eval (batched) ──────────────────────────────────────
    print("\n--- PROMPT EVAL (batched) ---", flush=True)
    lib.llama_perf_context_reset(ctx)

    # Debug: verify batch struct
    print(f"    sizeof(llama_batch) = {ctypes.sizeof(llama_batch)} (Python)", flush=True)
    batch = lib.llama_batch_get_one(tokens, n_tokens)
    print(f"    batch.n_tokens = {batch.n_tokens}", flush=True)
    print(f"    batch.token addr = {ctypes.addressof(batch.token.contents) if batch.token else 'NULL'}", flush=True)
    print(f"    batch.embd = {batch.embd}", flush=True)
    print(f"    batch.pos = {ctypes.cast(batch.pos, ctypes.c_void_p).value if batch.pos else 'NULL'}", flush=True)
    print(f"    batch.logits = {ctypes.cast(batch.logits, ctypes.c_void_p).value if batch.logits else 'NULL'}", flush=True)
    if batch.token:
        print(f"    batch.token[0] = {batch.token[0]}", flush=True)
        print(f"    batch.token[1] = {batch.token[1]}", flush=True)
    print(f"    Calling llama_decode...", flush=True)

    t0 = time.perf_counter()
    rc = lib.llama_decode(ctx, batch)
    lib.llama_synchronize(ctx)
    t_prompt = time.perf_counter() - t0

    if rc != 0:
        print(f"FATAL: prompt decode failed with rc={rc}")
        lib.llama_free(ctx)
        lib.llama_model_free(model)
        sys.exit(1)

    prompt_tps = n_tokens / t_prompt
    print(f"    {n_tokens} tokens in {t_prompt*1000:.1f}ms = {prompt_tps:.1f} tok/s")

    # Get first token via argmax
    logits_ptr = lib.llama_get_logits_ith(ctx, -1)
    if not logits_ptr:
        print("FATAL: no logits!")
        sys.exit(1)
    first_token = argmax_logits(logits_ptr, n_vocab)
    print(f"    First token: {first_token} = '{token_to_str(lib, vocab, first_token)}'")

    # ─── Phase 2: Token-by-token decode with detailed timing ─────────────────
    print(f"\n--- DECODE: {n_gen_tokens} tokens, one at a time ---")
    print(f"{'Tok#':>4} {'Total(ms)':>10} {'Decode(ms)':>11} {'Sync(ms)':>9} {'Logits(ms)':>11} {'tok/s':>7}  Text")
    print("-" * 78)

    generated = [first_token]
    decode_times = []
    sync_times = []
    logit_times = []
    total_times = []
    output_text = token_to_str(lib, vocab, first_token)

    cur_token = first_token

    for i in range(n_gen_tokens):
        if cur_token == eos_token:
            print(f"  [EOS at token {i}]")
            break

        # Prepare single-token batch
        tok_arr = (ctypes.c_int32 * 1)(cur_token)

        t_total_start = time.perf_counter()

        # Step 1: decode
        batch = lib.llama_batch_get_one(tok_arr, 1)
        t_decode_start = time.perf_counter()
        rc = lib.llama_decode(ctx, batch)
        t_decode_end = time.perf_counter()

        if rc != 0:
            print(f"  FATAL: decode failed at token {i} with rc={rc}")
            break

        # Step 2: synchronize (wait for GPU)
        t_sync_start = time.perf_counter()
        lib.llama_synchronize(ctx)
        t_sync_end = time.perf_counter()

        # Step 3: read logits + argmax
        t_logit_start = time.perf_counter()
        logits_ptr = lib.llama_get_logits_ith(ctx, -1)
        next_token = argmax_logits(logits_ptr, n_vocab)
        t_logit_end = time.perf_counter()

        t_total_end = time.perf_counter()

        # Record
        dt_decode = (t_decode_end - t_decode_start) * 1000
        dt_sync   = (t_sync_end - t_sync_start) * 1000
        dt_logit  = (t_logit_end - t_logit_start) * 1000
        dt_total  = (t_total_end - t_total_start) * 1000

        decode_times.append(dt_decode)
        sync_times.append(dt_sync)
        logit_times.append(dt_logit)
        total_times.append(dt_total)

        piece = token_to_str(lib, vocab, next_token)
        output_text += piece
        tps = 1000.0 / dt_total if dt_total > 0 else 0

        print(f"{i+1:>4} {dt_total:>10.2f} {dt_decode:>11.2f} {dt_sync:>9.2f} {dt_logit:>11.3f} {tps:>7.1f}  {repr(piece)}")

        generated.append(next_token)
        cur_token = next_token

    # ─── Summary ──────────────────────────────────────────────────────────────
    n_decoded = len(total_times)
    if n_decoded > 0:
        print()
        print("=" * 70)
        print("DIRECT API RESULTS SUMMARY")
        print("=" * 70)

        avg_total  = sum(total_times) / n_decoded
        avg_decode = sum(decode_times) / n_decoded
        avg_sync   = sum(sync_times) / n_decoded
        avg_logit  = sum(logit_times) / n_decoded

        if n_decoded > 2:
            warm_total  = sum(total_times[1:]) / (n_decoded - 1)
            warm_decode = sum(decode_times[1:]) / (n_decoded - 1)
            warm_sync   = sum(sync_times[1:]) / (n_decoded - 1)
        else:
            warm_total = avg_total
            warm_decode = avg_decode
            warm_sync = avg_sync

        print(f"\n  Tokens decoded: {n_decoded}")
        print(f"  Prompt eval:    {prompt_tps:.1f} tok/s ({t_prompt*1000:.1f}ms for {n_tokens} tokens)")
        print(f"\n  Per-token averages (ALL):")
        print(f"    Total:     {avg_total:.2f} ms  ({1000/avg_total:.1f} tok/s)")
        print(f"    Decode:    {avg_decode:.2f} ms")
        print(f"    Sync:      {avg_sync:.2f} ms")
        print(f"    Logits:    {avg_logit:.3f} ms")
        print(f"\n  Per-token averages (WARM, skip first):")
        print(f"    Total:     {warm_total:.2f} ms  ({1000/warm_total:.1f} tok/s)")
        print(f"    Decode:    {warm_decode:.2f} ms")
        print(f"    Sync:      {warm_sync:.2f} ms")
        print(f"\n  First token:  {total_times[0]:.2f} ms ({1000/total_times[0]:.1f} tok/s)")
        if n_decoded > 1:
            print(f"  Second token: {total_times[1]:.2f} ms ({1000/total_times[1]:.1f} tok/s)")
        print(f"  Min token:    {min(total_times):.2f} ms ({1000/min(total_times):.1f} tok/s)")
        print(f"  Max token:    {max(total_times):.2f} ms ({1000/max(total_times):.1f} tok/s)")

        print(f"\n  Time breakdown (warm averages):")
        print(f"    Decode call: {warm_decode/warm_total*100:.1f}%")
        print(f"    GPU sync:    {warm_sync/warm_total*100:.1f}%")
        print(f"    Logit read:  {avg_logit/warm_total*100:.1f}%")

        print(f"\n  Generated text: {repr(output_text[:200])}")
    else:
        avg_total = warm_total = 0

    # Print llama.cpp internal perf counters
    print(f"\n  llama.cpp internal perf:")
    lib.llama_perf_context_print(ctx)

    # ─── Phase 3: Back-to-back decode (weight warmth test) ────────────────────
    if n_decoded > 0 and cur_token != eos_token:
        print("\n" + "=" * 70)
        print("BACK-TO-BACK DECODE TEST (weights warm in cache)")
        print("=" * 70)
        print("  Two rapid decodes — second should be faster (weights in page cache)")

        for run in range(2):
            tok_arr = (ctypes.c_int32 * 1)(cur_token)
            batch = lib.llama_batch_get_one(tok_arr, 1)

            t0 = time.perf_counter()
            rc = lib.llama_decode(ctx, batch)
            lib.llama_synchronize(ctx)
            t1 = time.perf_counter()

            logits_ptr = lib.llama_get_logits_ith(ctx, -1)
            next_token = argmax_logits(logits_ptr, n_vocab)
            t2 = time.perf_counter()

            dt_compute = (t1 - t0) * 1000
            dt_total   = (t2 - t0) * 1000
            tps = 1000.0 / dt_total if dt_total > 0 else 0

            label = "FIRST " if run == 0 else "SECOND"
            print(f"  {label}: {dt_total:.2f}ms total ({dt_compute:.2f}ms compute) = {tps:.1f} tok/s  token='{token_to_str(lib, vocab, next_token)}'")
            cur_token = next_token

        # ─── Phase 4: Batch-of-2 test ─────────────────────────────────────────
        print("\n" + "=" * 70)
        print("BATCH-OF-2 TEST (two tokens in one decode call)")
        print("=" * 70)

        try:
            # Use llama_batch_get_one for 2 consecutive tokens
            # This is simpler and lets the library handle positions
            tok_pair = (ctypes.c_int32 * 2)(cur_token, generated[-2] if len(generated) >= 2 else bos_token)
            batch2 = lib.llama_batch_get_one(tok_pair, 2)

            t0 = time.perf_counter()
            rc = lib.llama_decode(ctx, batch2)
            lib.llama_synchronize(ctx)
            t1 = time.perf_counter()
            dt = (t1 - t0) * 1000
            per_tok = dt / 2
            tps = 2000.0 / dt if dt > 0 else 0
            print(f"  2 tokens in {dt:.2f}ms = {per_tok:.2f}ms/tok = {tps:.1f} tok/s")
            if warm_total > 0:
                print(f"  (vs single-token warm avg: {warm_total:.2f}ms/tok)")
                if per_tok < warm_total:
                    print(f"  SPEEDUP: {warm_total/per_tok:.2f}x from batching!")
                else:
                    print(f"  No speedup from batch-of-2 (overhead is per-decode, not per-token)")
        except Exception as e:
            print(f"  Batch-of-2 test failed: {e}")

    # ─── Cleanup ──────────────────────────────────────────────────────────────
    lib.llama_free(ctx)
    lib.llama_model_free(model)
    lib.llama_backend_free()

    return {
        "prompt_tps": prompt_tps if n_decoded > 0 else 0,
        "avg_total_ms": avg_total if n_decoded > 0 else 0,
        "warm_total_ms": warm_total if n_decoded > 0 else 0,
        "warm_tps": 1000 / warm_total if n_decoded > 0 and warm_total > 0 else 0,
        "total_times": total_times,
    }


# ─── HTTP SERVER COMPARISON ──────────────────────────────────────────────────

def run_http_comparison(n_tokens=20):
    """Compare against HTTP server if it's running on port 8081."""
    print("\n" + "=" * 70)
    print("HTTP SERVER COMPARISON (port 8081)")
    print("=" * 70)

    try:
        import httpx
    except ImportError:
        print("  httpx not installed, skipping HTTP comparison")
        return None

    try:
        client = httpx.Client(base_url="http://127.0.0.1:8081", timeout=120)
        resp = client.get("/health")
        if resp.status_code != 200:
            print("  Server not healthy, skipping")
            return None
    except Exception as e:
        print(f"  Server not running on :8081 ({e}), skipping")
        return None

    print("  Server is running. Generating tokens via HTTP...")

    prompt = "Write a Python function to compute fibonacci numbers efficiently."

    # Warmup
    client.post("/v1/chat/completions", json={
        "model": "gemma",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    })

    import json as json_mod

    t0 = time.perf_counter()
    token_times = []
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "gemma",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }) as resp:
        last_t = t0
        for line in resp.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                now = time.perf_counter()
                token_times.append((now - last_t) * 1000)
                last_t = now

    t_total = time.perf_counter() - t0

    if len(token_times) > 2:
        ttft = token_times[0]
        decode_times = token_times[1:]
        avg_decode = sum(decode_times) / len(decode_times)
        http_tps = 1000.0 / avg_decode if avg_decode > 0 else 0

        print(f"  TTFT: {ttft:.1f}ms")
        print(f"  Decode tokens: {len(decode_times)}")
        print(f"  Avg per-token: {avg_decode:.2f}ms = {http_tps:.1f} tok/s")
        print(f"  Total: {t_total*1000:.0f}ms")

        return {"ttft": ttft, "avg_decode_ms": avg_decode, "tps": http_tps}
    else:
        print("  Too few tokens received")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def check_server_running():
    """Check if llama-server is running on port 8081."""
    try:
        import httpx
        resp = httpx.get("http://127.0.0.1:8081/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def stop_server():
    """Stop the running llama-server process."""
    import subprocess
    print("  Stopping llama-server...")
    subprocess.run(["pkill", "-f", "llama-server.*8081"], capture_output=True)
    # Wait for it to actually stop
    import time
    for _ in range(30):
        time.sleep(0.5)
        if not check_server_running():
            print("  Server stopped.")
            return True
    print("  WARNING: Server may still be running!")
    return False


def restart_server():
    """Restart the llama-server with standard turbo config."""
    import subprocess
    print("  Restarting llama-server...")
    subprocess.Popen([
        "/Users/marcsolomon/llama-cpp-turboquant/build/bin/llama-server",
        "--model", MODEL_PATH,
        "--port", "8081",
        "-ngl", "999", "--mmap",
        "-ctk", "q8_0", "-ctv", "turbo4",
        "-fa", "on", "-c", "32768",
        "--threads", "10", "-b", "2048", "-ub", "512",
        "-np", "1", "-fit", "off", "--cache-ram", "0",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Wait for it to come up
    import time
    for _ in range(60):
        time.sleep(1)
        if check_server_running():
            print("  Server restarted.")
            return True
    print("  WARNING: Server did not come back up!")
    return False


def main():
    parser = argparse.ArgumentParser(description="Direct llama.cpp C API decode benchmark")
    parser.add_argument("--tokens", type=int, default=30, help="Number of tokens to generate (default: 30)")
    parser.add_argument("--no-http", action="store_true", help="Skip HTTP server comparison")
    parser.add_argument("--no-turbo", action="store_true", help="Use q8_0 KV instead of TurboQuant")
    parser.add_argument("--no-restart", action="store_true", help="Don't restart server after test")
    args = parser.parse_args()

    # IMPORTANT: We can't have two model instances on 16GB GPU.
    # Strategy: collect HTTP baseline first, then stop server, run direct API.

    http_results = None
    server_was_running = check_server_running()

    if not args.no_http and server_was_running:
        http_results = run_http_comparison(n_tokens=args.tokens)

    # Stop server to free GPU memory for direct API test
    if server_was_running:
        print("\n  NOTE: Stopping llama-server to free GPU memory for direct API test...")
        stop_server()
        import time
        time.sleep(2)  # Let GPU memory settle

    direct_results = run_direct_decode(n_gen_tokens=args.tokens, use_turboquant=not args.no_turbo)

    # Restart server if it was running
    if server_was_running and not args.no_restart:
        restart_server()

    # Final comparison
    if http_results and direct_results and direct_results["warm_tps"] > 0:
        print("\n" + "=" * 70)
        print("FINAL COMPARISON: Direct API vs HTTP Server")
        print("=" * 70)
        d_tps = direct_results["warm_tps"]
        h_tps = http_results["tps"]
        print(f"  Direct API (warm): {d_tps:.1f} tok/s ({direct_results['warm_total_ms']:.2f} ms/tok)")
        print(f"  HTTP Server:       {h_tps:.1f} tok/s ({http_results['avg_decode_ms']:.2f} ms/tok)")
        if d_tps > 0 and h_tps > 0:
            speedup = d_tps / h_tps
            overhead = http_results['avg_decode_ms'] - direct_results['warm_total_ms']
            print(f"  Speedup:           {speedup:.2f}x")
            print(f"  HTTP overhead:     {overhead:.2f} ms/tok")
            if speedup > 1.05:
                print(f"  --> Direct API is {(speedup-1)*100:.0f}% FASTER")
            elif speedup < 0.95:
                print(f"  --> HTTP server is {(1/speedup-1)*100:.0f}% faster (unexpected!)")
            else:
                print(f"  --> Effectively the SAME speed (HTTP overhead is negligible)")

    print("\nDone.")


if __name__ == "__main__":
    main()
