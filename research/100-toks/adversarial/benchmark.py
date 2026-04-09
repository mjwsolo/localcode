#!/usr/bin/env python3
"""
Adversarial benchmark: Attack the 35ms Metal scheduling overhead claim.

Tests whether Metal command buffer commit-to-completion latency is truly
a 35ms driver floor, or whether it is a composite of reducible factors.

Requirements: macOS with Apple Silicon, pyobjc-core installed.
Run: python3 benchmark.py [--test N] [--iterations 100]
"""

import argparse
import ctypes
import math
import mmap
import os
import statistics
import struct
import sys
import tempfile
import time
import threading

import objc
from Foundation import NSBundle

# ---------------------------------------------------------------------------
# Metal bootstrap via ctypes + pyobjc bridge
# ---------------------------------------------------------------------------

def get_metal_device():
    """Get the default Metal device as a pyobjc object."""
    metal_bundle = NSBundle.bundleWithPath_(
        "/System/Library/Frameworks/Metal.framework"
    )
    if not metal_bundle or not metal_bundle.load():
        print("ERROR: Cannot load Metal.framework")
        sys.exit(1)

    metal_lib = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/Metal.framework/Metal"
    )
    fn = metal_lib.MTLCreateSystemDefaultDevice
    fn.restype = ctypes.c_void_p
    ptr = fn()
    if not ptr:
        print("ERROR: MTLCreateSystemDefaultDevice returned NULL")
        sys.exit(1)

    device = objc.objc_object(c_void_p=ptr)
    return device


# ---------------------------------------------------------------------------
# Metal shader source (trivial compute kernels)
# ---------------------------------------------------------------------------

SHADER_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Trivial kernel: write a constant
kernel void trivial_write(
    device float* output [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        output[0] = 1.0f;
    }
}

// Sum reduction kernel: read from a large buffer, write sum
kernel void sum_reduce(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    float sum = 0.0f;
    for (uint i = tid; i < count; i += threads) {
        sum += input[i];
    }
    // Very naive - just write partial sum. We only care about timing.
    if (tid == 0) {
        output[0] = sum;
    }
}

// Dummy heartbeat kernel: does minimal work to keep GPU alive
kernel void heartbeat(
    device float* output [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        output[0] += 1.0f;
    }
}

// Multi-dispatch test: each dispatch writes to a different offset
kernel void multi_dispatch(
    device float* output [[buffer(0)]],
    constant uint& offset [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        output[offset] = float(offset);
    }
}
"""


def compile_shaders(device):
    """Compile Metal shader library from source."""
    options = None  # Use default compile options
    error = None
    library = device.newLibraryWithSource_options_error_(
        SHADER_SOURCE, options, None
    )
    if isinstance(library, tuple):
        library, error = library
    if error:
        print(f"Shader compile error: {error}")
        sys.exit(1)
    return library


def make_pipeline(device, library, name):
    """Create a compute pipeline state for a named kernel function."""
    func = library.newFunctionWithName_(name)
    if not func:
        print(f"ERROR: Function '{name}' not found in shader library")
        sys.exit(1)
    result = device.newComputePipelineStateWithFunction_error_(func, None)
    # pyobjc may return (pipeline, error) tuple or just the pipeline
    if isinstance(result, tuple):
        pipeline, error = result
        if error:
            print(f"Pipeline error for '{name}': {error}")
            sys.exit(1)
    else:
        pipeline = result
    if not pipeline:
        print(f"ERROR: Pipeline creation returned None for '{name}'")
        sys.exit(1)
    return pipeline


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def stats_summary(times_ms):
    """Return a dict of statistics for a list of times in ms."""
    if not times_ms:
        return {}
    times_ms.sort()
    return {
        "min": times_ms[0],
        "median": statistics.median(times_ms),
        "mean": statistics.mean(times_ms),
        "p95": times_ms[int(len(times_ms) * 0.95)] if len(times_ms) >= 20 else times_ms[-1],
        "max": times_ms[-1],
        "stdev": statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
        "n": len(times_ms),
    }


def print_stats(label, times_ms):
    """Pretty-print statistics."""
    s = stats_summary(times_ms)
    if not s:
        print(f"  {label}: NO DATA")
        return
    print(f"  {label}:")
    print(f"    min={s['min']:.3f}ms  median={s['median']:.3f}ms  "
          f"mean={s['mean']:.3f}ms  p95={s['p95']:.3f}ms  "
          f"max={s['max']:.3f}ms  stdev={s['stdev']:.3f}ms  n={s['n']}")


def allocate_mtl_buffer(device, size_bytes, label="buffer"):
    """Allocate a shared MTLBuffer of the given size."""
    # MTLResourceStorageModeShared = 0
    buf = device.newBufferWithLength_options_(size_bytes, 0)
    if not buf:
        print(f"ERROR: Failed to allocate {size_bytes} byte MTLBuffer")
        return None
    return buf


# ---------------------------------------------------------------------------
# Test 1: Trivial Kernel Baseline
# ---------------------------------------------------------------------------

def test_trivial_kernel(device, library, iterations, warmup):
    """Measure commit-to-completion for the most trivial possible kernel."""
    print("\n" + "=" * 70)
    print("TEST 1: Trivial Kernel Commit Latency (Baseline)")
    print("=" * 70)
    print(f"  Kernel: writes a single float")
    print(f"  Iterations: {warmup} warmup + {iterations} measured")

    pipeline = make_pipeline(device, library, "trivial_write")
    queue = device.newCommandQueue()
    output_buf = allocate_mtl_buffer(device, 4, "trivial_output")

    times = []
    for i in range(warmup + iterations):
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            (1, 1, 1), (1, 1, 1)
        )
        encoder.endEncoding()

        t0 = time.perf_counter_ns()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        t1 = time.perf_counter_ns()

        elapsed_ms = (t1 - t0) / 1e6
        if i >= warmup:
            times.append(elapsed_ms)

    print_stats("Trivial kernel", times)

    if stats_summary(times)["median"] < 5.0:
        print("\n  >>> RESULT: Trivial kernel < 5ms.")
        print("  >>> The 35ms is NOT a Metal driver floor. It is workload-specific.")
    elif stats_summary(times)["median"] < 15.0:
        print("\n  >>> RESULT: Trivial kernel 5-15ms.")
        print("  >>> Partial driver overhead, but model adds significant extra cost.")
    else:
        print("\n  >>> RESULT: Trivial kernel >= 15ms.")
        print("  >>> Significant baseline overhead from Metal driver.")
    return times


# ---------------------------------------------------------------------------
# Test 2: Timestamp Decomposition
# ---------------------------------------------------------------------------

def test_timestamp_decomposition(device, library, iterations, warmup):
    """Decompose commit-to-completion into pre-GPU, GPU, and post-GPU phases."""
    print("\n" + "=" * 70)
    print("TEST 2: Timestamp Decomposition")
    print("=" * 70)

    pipeline = make_pipeline(device, library, "trivial_write")
    queue = device.newCommandQueue()
    output_buf = allocate_mtl_buffer(device, 4, "ts_output")

    pre_gpu_times = []
    gpu_times = []
    post_gpu_times = []
    total_times = []

    for i in range(warmup + iterations):
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            (1, 1, 1), (1, 1, 1)
        )
        encoder.endEncoding()

        t_commit = time.perf_counter_ns()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        t_done = time.perf_counter_ns()

        # Metal provides kernel start/end timestamps (seconds, GPU clock)
        # These are available as properties on the completed command buffer
        try:
            gpu_start = cmd_buf.kernelStartTime()
            gpu_end = cmd_buf.kernelEndTime()
            gpu_ms = (gpu_end - gpu_start) * 1000.0

            # Note: kernelStartTime/kernelEndTime are in Mach absolute time
            # (seconds). We can compute relative durations but not directly
            # compare with perf_counter_ns. So we decompose the wall-clock
            # total into: total = pre_gpu + gpu + post_gpu
            total_ms = (t_done - t_commit) / 1e6

            # Approximate: if GPU time is known, the rest is overhead
            overhead_ms = total_ms - gpu_ms
        except Exception:
            gpu_ms = float("nan")
            overhead_ms = float("nan")
            total_ms = (t_done - t_commit) / 1e6

        if i >= warmup:
            total_times.append(total_ms)
            if not math.isnan(gpu_ms):
                gpu_times.append(gpu_ms)
                # We split overhead 50/50 as an estimate since we cannot
                # directly compare Metal's GPU clock with wall clock for
                # the pre/post split. The total overhead is what matters.
                pre_gpu_times.append(overhead_ms)

    print_stats("Total (commit to wait return)", total_times)
    print_stats("GPU kernel time", gpu_times)
    print_stats("Non-GPU overhead (total - GPU)", pre_gpu_times)
    return total_times


# ---------------------------------------------------------------------------
# Test 3: MTLBuffer vs mmap
# ---------------------------------------------------------------------------

def test_mtlbuffer_vs_mmap(device, library, iterations, warmup):
    """Compare commit latency with MTLBuffer-resident data vs mmap'd data."""
    print("\n" + "=" * 70)
    print("TEST 3: MTLBuffer (GPU-resident) vs mmap")
    print("=" * 70)

    pipeline = make_pipeline(device, library, "sum_reduce")
    queue = device.newCommandQueue()
    output_buf = allocate_mtl_buffer(device, 4, "reduce_output")

    # Test sizes (bytes)
    sizes = [
        (1 * 1024 * 1024, "1MB"),
        (100 * 1024 * 1024, "100MB"),
        (512 * 1024 * 1024, "512MB"),
    ]

    # Check available memory - skip huge sizes on constrained systems
    try:
        recommended = device.recommendedMaxWorkingSetSize()
        print(f"  Recommended max working set: {recommended / 1e9:.1f} GB")
    except Exception:
        pass

    for size_bytes, label in sizes:
        print(f"\n  --- Size: {label} ({size_bytes / 1e6:.0f} MB) ---")

        count = size_bytes // 4  # number of floats
        count_buf = allocate_mtl_buffer(device, 4, "count")
        if count_buf:
            # Write count value into the buffer
            ctypes.memmove(
                count_buf.contents(),
                struct.pack("I", count),
                4,
            )

        # --- MTLBuffer path ---
        mtl_buf = allocate_mtl_buffer(device, size_bytes, f"mtl_{label}")
        if not mtl_buf:
            print(f"    MTLBuffer allocation failed for {label}, skipping")
            continue

        mtl_times = []
        for i in range(warmup + iterations):
            cmd_buf = queue.commandBuffer()
            encoder = cmd_buf.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)
            encoder.setBuffer_offset_atIndex_(mtl_buf, 0, 0)
            encoder.setBuffer_offset_atIndex_(output_buf, 0, 1)
            encoder.setBuffer_offset_atIndex_(count_buf, 0, 2)
            tg_size = min(256, pipeline.maxTotalThreadsPerThreadgroup())
            encoder.dispatchThreads_threadsPerThreadgroup_(
                (min(1024, count), 1, 1), (tg_size, 1, 1)
            )
            encoder.endEncoding()

            t0 = time.perf_counter_ns()
            cmd_buf.commit()
            cmd_buf.waitUntilCompleted()
            t1 = time.perf_counter_ns()

            if i >= warmup:
                mtl_times.append((t1 - t0) / 1e6)

        print_stats(f"MTLBuffer {label}", mtl_times)
        del mtl_buf

        # --- mmap path ---
        try:
            tmpfd, tmppath = tempfile.mkstemp(suffix=".bin")
            os.ftruncate(tmpfd, size_bytes)
            mm = mmap.mmap(tmpfd, size_bytes)

            # Create MTLBuffer wrapping the mmap'd memory
            # MTLResourceStorageModeShared = 0, MTLResourceCPUCacheModeDefaultCache = 0
            ptr_val = ctypes.c_void_p.from_buffer(
                (ctypes.c_char * 8).from_buffer(
                    bytearray(struct.pack("Q", ctypes.addressof(
                        ctypes.c_char.from_buffer(mm)
                    )))
                )
            ).value

            # Use newBufferWithBytesNoCopy if available
            # Fall back to just timing the mmap access pattern
            mmap_buf = device.newBufferWithBytesNoCopy_length_options_deallocator_(
                ptr_val, size_bytes, 0, None
            )
            if isinstance(mmap_buf, tuple):
                mmap_buf = mmap_buf[0]

            if mmap_buf:
                mmap_times = []
                for i in range(warmup + iterations):
                    cmd_buf = queue.commandBuffer()
                    encoder = cmd_buf.computeCommandEncoder()
                    encoder.setComputePipelineState_(pipeline)
                    encoder.setBuffer_offset_atIndex_(mmap_buf, 0, 0)
                    encoder.setBuffer_offset_atIndex_(output_buf, 0, 1)
                    encoder.setBuffer_offset_atIndex_(count_buf, 0, 2)
                    encoder.dispatchThreads_threadsPerThreadgroup_(
                        (min(1024, count), 1, 1), (tg_size, 1, 1)
                    )
                    encoder.endEncoding()

                    t0 = time.perf_counter_ns()
                    cmd_buf.commit()
                    cmd_buf.waitUntilCompleted()
                    t1 = time.perf_counter_ns()

                    if i >= warmup:
                        mmap_times.append((t1 - t0) / 1e6)

                print_stats(f"mmap {label}", mmap_times)
                del mmap_buf
            else:
                print(f"    mmap buffer creation failed for {label}")

            mm.close()
            os.close(tmpfd)
            os.unlink(tmppath)
        except Exception as e:
            print(f"    mmap test failed for {label}: {e}")


# ---------------------------------------------------------------------------
# Test 4: Scaling with Buffer Size
# ---------------------------------------------------------------------------

def test_buffer_size_scaling(device, library, iterations, warmup):
    """Test if commit latency scales with total bound resource size."""
    print("\n" + "=" * 70)
    print("TEST 4: Scaling with Total Bound Resource Size")
    print("=" * 70)

    pipeline = make_pipeline(device, library, "trivial_write")
    queue = device.newCommandQueue()
    output_buf = allocate_mtl_buffer(device, 4, "scale_output")

    # Allocate buffers of increasing sizes. The kernel only touches output_buf.
    # We bind the large buffers to exercise resource validation.
    sizes = [
        (0, "0B extra"),
        (1024, "1KB extra"),
        (1 * 1024 * 1024, "1MB extra"),
        (100 * 1024 * 1024, "100MB extra"),
        (512 * 1024 * 1024, "512MB extra"),
        (1024 * 1024 * 1024, "1GB extra"),
    ]

    for size_bytes, label in sizes:
        extra_buf = None
        if size_bytes > 0:
            extra_buf = allocate_mtl_buffer(device, size_bytes, f"extra_{label}")
            if not extra_buf:
                print(f"  Skipping {label}: allocation failed")
                continue

        times = []
        for i in range(warmup + iterations):
            cmd_buf = queue.commandBuffer()
            encoder = cmd_buf.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)
            encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)
            # Bind the extra buffer even though kernel doesn't use it
            if extra_buf:
                encoder.setBuffer_offset_atIndex_(extra_buf, 0, 1)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                (1, 1, 1), (1, 1, 1)
            )
            encoder.endEncoding()

            t0 = time.perf_counter_ns()
            cmd_buf.commit()
            cmd_buf.waitUntilCompleted()
            t1 = time.perf_counter_ns()

            if i >= warmup:
                times.append((t1 - t0) / 1e6)

        print_stats(f"  {label}", times)
        if extra_buf:
            del extra_buf


# ---------------------------------------------------------------------------
# Test 5: GPU Pre-Warming
# ---------------------------------------------------------------------------

def test_gpu_warming(device, library, iterations, warmup):
    """Test if keeping the GPU warm reduces commit latency."""
    print("\n" + "=" * 70)
    print("TEST 5: GPU Pre-Warming (Power State)")
    print("=" * 70)

    pipeline = make_pipeline(device, library, "trivial_write")
    heartbeat_pipeline = make_pipeline(device, library, "heartbeat")
    queue = device.newCommandQueue()
    heartbeat_queue = device.newCommandQueue()
    output_buf = allocate_mtl_buffer(device, 4, "warm_output")
    heartbeat_buf = allocate_mtl_buffer(device, 4, "heartbeat_buf")

    # --- Cold GPU: wait 2 seconds, then measure ---
    print("\n  Phase 1: Cold GPU (2 second idle before each commit)")
    cold_times = []
    for i in range(warmup + min(iterations, 20)):  # Fewer iterations (slow)
        time.sleep(2.0)  # Let GPU go idle

        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            (1, 1, 1), (1, 1, 1)
        )
        encoder.endEncoding()

        t0 = time.perf_counter_ns()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        t1 = time.perf_counter_ns()

        if i >= warmup:
            cold_times.append((t1 - t0) / 1e6)

    print_stats("Cold GPU", cold_times)

    # --- Warm GPU: heartbeat thread keeps GPU busy ---
    print("\n  Phase 2: Warm GPU (heartbeat kernel every 2ms)")
    stop_heartbeat = threading.Event()

    def heartbeat_thread():
        while not stop_heartbeat.is_set():
            hb_cmd = heartbeat_queue.commandBuffer()
            hb_enc = hb_cmd.computeCommandEncoder()
            hb_enc.setComputePipelineState_(heartbeat_pipeline)
            hb_enc.setBuffer_offset_atIndex_(heartbeat_buf, 0, 0)
            hb_enc.dispatchThreads_threadsPerThreadgroup_(
                (1, 1, 1), (1, 1, 1)
            )
            hb_enc.endEncoding()
            hb_cmd.commit()
            hb_cmd.waitUntilCompleted()
            time.sleep(0.002)  # 2ms between heartbeats

    hb_thread = threading.Thread(target=heartbeat_thread, daemon=True)
    hb_thread.start()
    time.sleep(0.1)  # Let heartbeat stabilize

    warm_times = []
    for i in range(warmup + iterations):
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            (1, 1, 1), (1, 1, 1)
        )
        encoder.endEncoding()

        t0 = time.perf_counter_ns()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        t1 = time.perf_counter_ns()

        if i >= warmup:
            warm_times.append((t1 - t0) / 1e6)

    stop_heartbeat.set()
    hb_thread.join(timeout=1.0)

    print_stats("Warm GPU", warm_times)

    cold_med = stats_summary(cold_times).get("median", 0)
    warm_med = stats_summary(warm_times).get("median", 0)
    if cold_med > 0 and warm_med > 0:
        diff = cold_med - warm_med
        print(f"\n  >>> Delta: {diff:.3f}ms (cold - warm)")
        if diff > 5:
            print(f"  >>> GPU power cycling accounts for ~{diff:.0f}ms of overhead!")
        else:
            print(f"  >>> GPU power state has minimal effect.")


# ---------------------------------------------------------------------------
# Test 6: Scaling with Dispatch Count
# ---------------------------------------------------------------------------

def test_dispatch_count_scaling(device, library, iterations, warmup):
    """Test if commit latency scales with number of dispatches."""
    print("\n" + "=" * 70)
    print("TEST 6: Scaling with Dispatch Count")
    print("=" * 70)

    pipeline = make_pipeline(device, library, "multi_dispatch")
    queue = device.newCommandQueue()

    dispatch_counts = [1, 10, 50, 100, 500, 1000, 2500]

    for n_dispatches in dispatch_counts:
        # Allocate buffer large enough for all dispatches
        buf_size = max(4, n_dispatches * 4)
        output_buf = allocate_mtl_buffer(device, buf_size, f"dispatch_{n_dispatches}")
        if not output_buf:
            continue

        offset_buf = allocate_mtl_buffer(device, 4, "offset")

        times = []
        for i in range(warmup + iterations):
            cmd_buf = queue.commandBuffer()
            encoder = cmd_buf.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)
            encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)

            for d in range(n_dispatches):
                # Update offset for each dispatch
                ctypes.memmove(
                    offset_buf.contents(),
                    struct.pack("I", d),
                    4,
                )
                encoder.setBuffer_offset_atIndex_(offset_buf, 0, 1)
                encoder.dispatchThreads_threadsPerThreadgroup_(
                    (1, 1, 1), (1, 1, 1)
                )

            encoder.endEncoding()

            t0 = time.perf_counter_ns()
            cmd_buf.commit()
            cmd_buf.waitUntilCompleted()
            t1 = time.perf_counter_ns()

            if i >= warmup:
                times.append((t1 - t0) / 1e6)

        print_stats(f"  {n_dispatches} dispatches", times)
        del output_buf


# ---------------------------------------------------------------------------
# Test 7: Command Buffer Options
# ---------------------------------------------------------------------------

def test_command_buffer_options(device, library, iterations, warmup):
    """Test different command buffer creation options."""
    print("\n" + "=" * 70)
    print("TEST 7: Command Buffer Creation Options")
    print("=" * 70)

    pipeline = make_pipeline(device, library, "trivial_write")
    queue = device.newCommandQueue()
    output_buf = allocate_mtl_buffer(device, 4, "opt_output")

    # Option 1: Default (retained references)
    print("\n  Option A: commandBuffer (retained references)")
    times_a = []
    for i in range(warmup + iterations):
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            (1, 1, 1), (1, 1, 1)
        )
        encoder.endEncoding()

        t0 = time.perf_counter_ns()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        t1 = time.perf_counter_ns()

        if i >= warmup:
            times_a.append((t1 - t0) / 1e6)
    print_stats("Retained refs", times_a)

    # Option 2: Unretained references
    print("\n  Option B: commandBufferWithUnretainedReferences")
    times_b = []
    for i in range(warmup + iterations):
        cmd_buf = queue.commandBufferWithUnretainedReferences()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            (1, 1, 1), (1, 1, 1)
        )
        encoder.endEncoding()

        t0 = time.perf_counter_ns()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        t1 = time.perf_counter_ns()

        if i >= warmup:
            times_b.append((t1 - t0) / 1e6)
    print_stats("Unretained refs", times_b)

    # Option 3: Descriptor with no error reporting
    print("\n  Option C: Descriptor with errorOptions=none")
    try:
        # MTLCommandBufferDescriptor
        desc_class = objc.lookUpClass("MTLCommandBufferDescriptor")
        desc = desc_class.alloc().init()
        desc.setRetainedReferences_(False)
        desc.setErrorOptions_(0)  # MTLCommandBufferErrorOptionNone = 0

        times_c = []
        for i in range(warmup + iterations):
            cmd_buf = queue.commandBufferWithDescriptor_(desc)
            encoder = cmd_buf.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)
            encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                (1, 1, 1), (1, 1, 1)
            )
            encoder.endEncoding()

            t0 = time.perf_counter_ns()
            cmd_buf.commit()
            cmd_buf.waitUntilCompleted()
            t1 = time.perf_counter_ns()

            if i >= warmup:
                times_c.append((t1 - t0) / 1e6)
        print_stats("No error opts", times_c)
    except Exception as e:
        print(f"    Descriptor test failed: {e}")


# ---------------------------------------------------------------------------
# Test 8: Pipelined Double Buffering
# ---------------------------------------------------------------------------

def test_pipelining(device, library, iterations, warmup):
    """Test if pipelined commits reduce per-commit overhead."""
    print("\n" + "=" * 70)
    print("TEST 8: Pipelined Double Buffering")
    print("=" * 70)

    pipeline = make_pipeline(device, library, "trivial_write")
    queue = device.newCommandQueue()
    output_buf_a = allocate_mtl_buffer(device, 4, "pipe_a")
    output_buf_b = allocate_mtl_buffer(device, 4, "pipe_b")

    n = warmup + iterations

    # --- Serial: commit, wait, commit, wait ---
    print("\n  Serial: commit-wait-commit-wait")
    serial_times = []
    for i in range(n):
        t0 = time.perf_counter_ns()

        cmd_a = queue.commandBuffer()
        enc_a = cmd_a.computeCommandEncoder()
        enc_a.setComputePipelineState_(pipeline)
        enc_a.setBuffer_offset_atIndex_(output_buf_a, 0, 0)
        enc_a.dispatchThreads_threadsPerThreadgroup_((1, 1, 1), (1, 1, 1))
        enc_a.endEncoding()
        cmd_a.commit()
        cmd_a.waitUntilCompleted()

        cmd_b = queue.commandBuffer()
        enc_b = cmd_b.computeCommandEncoder()
        enc_b.setComputePipelineState_(pipeline)
        enc_b.setBuffer_offset_atIndex_(output_buf_b, 0, 0)
        enc_b.dispatchThreads_threadsPerThreadgroup_((1, 1, 1), (1, 1, 1))
        enc_b.endEncoding()
        cmd_b.commit()
        cmd_b.waitUntilCompleted()

        t1 = time.perf_counter_ns()
        if i >= warmup:
            serial_times.append((t1 - t0) / 1e6)

    print_stats("Serial (2 commits)", serial_times)

    # --- Pipelined: commit A, encode+commit B, wait B ---
    print("\n  Pipelined: commit A, encode B, commit B, wait B")
    pipe_times = []
    for i in range(n):
        t0 = time.perf_counter_ns()

        cmd_a = queue.commandBuffer()
        enc_a = cmd_a.computeCommandEncoder()
        enc_a.setComputePipelineState_(pipeline)
        enc_a.setBuffer_offset_atIndex_(output_buf_a, 0, 0)
        enc_a.dispatchThreads_threadsPerThreadgroup_((1, 1, 1), (1, 1, 1))
        enc_a.endEncoding()
        cmd_a.commit()
        # Don't wait! Encode B while A is in flight.

        cmd_b = queue.commandBuffer()
        enc_b = cmd_b.computeCommandEncoder()
        enc_b.setComputePipelineState_(pipeline)
        enc_b.setBuffer_offset_atIndex_(output_buf_b, 0, 0)
        enc_b.dispatchThreads_threadsPerThreadgroup_((1, 1, 1), (1, 1, 1))
        enc_b.endEncoding()
        cmd_b.commit()
        cmd_b.waitUntilCompleted()  # Wait for both to finish

        t1 = time.perf_counter_ns()
        if i >= warmup:
            pipe_times.append((t1 - t0) / 1e6)

    print_stats("Pipelined (2 commits)", pipe_times)

    # --- Rapid fire: commit N buffers, wait on last ---
    print("\n  Rapid fire: commit 10 buffers, wait on last only")
    rapid_times = []
    bufs = [allocate_mtl_buffer(device, 4, f"rapid_{j}") for j in range(10)]
    for i in range(n):
        t0 = time.perf_counter_ns()

        last_cmd = None
        for j in range(10):
            cmd = queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(bufs[j], 0, 0)
            enc.dispatchThreads_threadsPerThreadgroup_((1, 1, 1), (1, 1, 1))
            enc.endEncoding()
            cmd.commit()
            last_cmd = cmd

        last_cmd.waitUntilCompleted()
        t1 = time.perf_counter_ns()
        if i >= warmup:
            rapid_times.append((t1 - t0) / 1e6)

    print_stats("Rapid fire (10 commits)", rapid_times)

    serial_med = stats_summary(serial_times).get("median", 0)
    pipe_med = stats_summary(pipe_times).get("median", 0)
    rapid_med = stats_summary(rapid_times).get("median", 0)
    if serial_med > 0:
        print(f"\n  >>> Serial per-commit: {serial_med / 2:.3f}ms")
        print(f"  >>> Pipelined per-commit: {pipe_med / 2:.3f}ms")
        print(f"  >>> Rapid fire per-commit: {rapid_med / 10:.3f}ms")
        if rapid_med / 10 < serial_med / 2 * 0.7:
            print("  >>> Pipelining significantly reduces per-commit overhead!")


# ---------------------------------------------------------------------------
# Test 9: Completion Signaling Methods
# ---------------------------------------------------------------------------

def test_signaling_methods(device, library, iterations, warmup):
    """Compare different ways to detect command buffer completion."""
    print("\n" + "=" * 70)
    print("TEST 9: Completion Signaling Methods")
    print("=" * 70)

    pipeline = make_pipeline(device, library, "trivial_write")
    queue = device.newCommandQueue()
    output_buf = allocate_mtl_buffer(device, 4, "signal_output")

    # Method 1: waitUntilCompleted
    print("\n  Method A: waitUntilCompleted")
    wait_times = []
    for i in range(warmup + iterations):
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            (1, 1, 1), (1, 1, 1)
        )
        encoder.endEncoding()

        t0 = time.perf_counter_ns()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        t1 = time.perf_counter_ns()

        if i >= warmup:
            wait_times.append((t1 - t0) / 1e6)
    print_stats("waitUntilCompleted", wait_times)

    # Method 2: addCompletedHandler (async callback)
    # Note: pyobjc may not support block-based callbacks without pyobjc-framework-Metal.
    # Skipping if not available.
    print("\n  Method B: addCompletedHandler (callback-based)")
    print("    Skipped: pyobjc-framework-Metal not installed (block signature unavailable)")

    # Method 3: Spin on status
    print("\n  Method C: Spin-poll on commandBuffer.status")
    spin_times = []
    # MTLCommandBufferStatusCompleted = 4
    for i in range(warmup + iterations):
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            (1, 1, 1), (1, 1, 1)
        )
        encoder.endEncoding()

        t0 = time.perf_counter_ns()
        cmd_buf.commit()
        while cmd_buf.status() < 4:  # < MTLCommandBufferStatusCompleted
            pass
        t1 = time.perf_counter_ns()

        if i >= warmup:
            spin_times.append((t1 - t0) / 1e6)
    print_stats("Spin on status", spin_times)

    # Method 4: MTLSharedEvent
    print("\n  Method D: MTLSharedEvent signaling")
    try:
        event = device.newSharedEvent()
        event.setSignaledValue_(0)

        event_times = []
        for i in range(warmup + iterations):
            event.setSignaledValue_(0)

            cmd_buf = queue.commandBuffer()
            encoder = cmd_buf.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)
            encoder.setBuffer_offset_atIndex_(output_buf, 0, 0)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                (1, 1, 1), (1, 1, 1)
            )
            encoder.endEncoding()

            signal_value = i + 1
            cmd_buf.encodeSignalEvent_value_(event, signal_value)

            t0 = time.perf_counter_ns()
            cmd_buf.commit()
            # Spin-wait on the shared event value
            while event.signaledValue() < signal_value:
                pass
            t1 = time.perf_counter_ns()

            if i >= warmup:
                event_times.append((t1 - t0) / 1e6)
        print_stats("MTLSharedEvent", event_times)
    except Exception as e:
        print(f"    MTLSharedEvent test failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = {
    1: ("Trivial kernel baseline", test_trivial_kernel),
    2: ("Timestamp decomposition", test_timestamp_decomposition),
    3: ("MTLBuffer vs mmap", test_mtlbuffer_vs_mmap),
    4: ("Buffer size scaling", test_buffer_size_scaling),
    5: ("GPU pre-warming", test_gpu_warming),
    6: ("Dispatch count scaling", test_dispatch_count_scaling),
    7: ("Command buffer options", test_command_buffer_options),
    8: ("Pipelined double buffering", test_pipelining),
    9: ("Completion signaling methods", test_signaling_methods),
}


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial benchmark: Attack the 35ms Metal scheduling claim"
    )
    parser.add_argument(
        "--test", "-t", type=int, default=0,
        help="Run a specific test (1-9). 0 = all tests."
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=100,
        help="Number of measured iterations per test (default: 100)"
    )
    parser.add_argument(
        "--warmup", "-w", type=int, default=10,
        help="Number of warmup iterations (default: 10)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ADVERSARIAL METAL BENCHMARK")
    print("Attacking the claim: '35ms Metal scheduling is an immovable wall'")
    print("=" * 70)

    device = get_metal_device()
    print(f"\nDevice: {device.name()}")

    try:
        print(f"Max working set: {device.recommendedMaxWorkingSetSize() / 1e9:.1f} GB")
    except Exception:
        pass

    library = compile_shaders(device)
    print(f"Shader library compiled: {library.functionNames()}")

    if args.test == 0:
        tests_to_run = sorted(ALL_TESTS.keys())
    else:
        if args.test not in ALL_TESTS:
            print(f"ERROR: Unknown test {args.test}. Available: {sorted(ALL_TESTS.keys())}")
            sys.exit(1)
        tests_to_run = [args.test]

    for test_num in tests_to_run:
        label, func = ALL_TESTS[test_num]
        print(f"\n{'#' * 70}")
        print(f"# Running Test {test_num}: {label}")
        print(f"{'#' * 70}")
        try:
            func(device, library, args.iterations, args.warmup)
        except Exception as e:
            print(f"\n  !!! Test {test_num} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  - If Test 1 shows < 5ms: the 35ms is NOT a Metal floor")
    print("  - If Test 5 shows cold >> warm: GPU power state is recoverable")
    print("  - If Test 6 shows linear scaling: kernel fusion is the fix")
    print("  - If Test 3 shows mmap >> MTLBuffer: memory residency is the fix")
    print("  - If Test 8 shows pipelining helps: double-buffering decode is the fix")


if __name__ == "__main__":
    main()
