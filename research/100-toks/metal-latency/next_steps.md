# Next Steps: Concrete Code Changes

## Phase 0: Measure (1-2 days)

Before changing anything, instrument the exact breakdown of the 35ms.

### Step 0.1: Detailed Timing Instrumentation

File: `ggml/src/ggml-metal/ggml-metal-context.m`

Add timing around every phase of `ggml_metal_graph_compute()` and `ggml_metal_synchronize()`:

```objc
// In ggml_metal_synchronize():
void ggml_metal_synchronize(ggml_metal_t ctx) {
    if (ctx->cmd_buf_last) {
        uint64_t t0 = mach_absolute_time();
        
        // Check if already completed
        MTLCommandBufferStatus status = [ctx->cmd_buf_last status];
        uint64_t t1 = mach_absolute_time();
        
        if (status < MTLCommandBufferStatusCompleted) {
            [ctx->cmd_buf_last waitUntilCompleted];
        }
        uint64_t t2 = mach_absolute_time();
        
        // Get GPU timestamps
        CFTimeInterval gpu_start = ctx->cmd_buf_last.GPUStartTime;
        CFTimeInterval gpu_end = ctx->cmd_buf_last.GPUEndTime;
        
        // Convert to ms
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        double status_ms = (t1 - t0) * info.numer / info.denom / 1e6;
        double wait_ms = (t2 - t1) * info.numer / info.denom / 1e6;
        double gpu_ms = (gpu_end - gpu_start) * 1000.0;
        
        static int decode_count = 0;
        if (++decode_count % 10 == 0) {
            GGML_LOG_INFO("sync: status_check=%.2fms wait=%.2fms gpu=%.2fms\n",
                          status_ms, wait_ms, gpu_ms);
        }
        
        ctx->cmd_buf_last = nil;
    }
}
```

### Step 0.2: Measure Split Overhead

Run with `GGML_SCHED_DEBUG=2` to see exactly why there are 2 graph splits:

```bash
GGML_SCHED_DEBUG=2 llama-server \
  --model <gguf-path> \
  -ngl 999 --mmap \
  -ctk q8_0 -ctv turbo4 \
  -fa on -c 32768 \
  --threads 10 -b 2048 -ub 512 \
  -np 1 -fit off --cache-ram 0
```

This will print which operations cause each split and what backend they're assigned to.

### Step 0.3: Test GPU Sleep Hypothesis

Add a busy-wait keep-alive between decode steps to prevent GPU sleep:

```objc
// After waitUntilCompleted, before returning from synchronize:
// Submit a tiny no-op to keep GPU clock up
{
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
    id<MTLCommandBuffer> noop = [queue commandBuffer];
    [noop commit];
    // Don't wait -- just keep the queue alive
}
```

If this alone improves decode speed by 5-15ms/token, GPU sleep is confirmed as
primary cause.


## Phase 1: Quick Wins (3-5 days)

### Step 1.1: GPU Keep-Alive via Continuous Queue

File: `ggml/src/ggml-metal/ggml-metal-context.m`

Add a mechanism to keep a no-op command buffer in the queue at all times during
active decode:

```objc
// New field in struct ggml_metal:
dispatch_source_t keepalive_timer;
bool decode_active;

// Start keepalive when decode begins
void ggml_metal_start_decode_session(ggml_metal_t ctx) {
    ctx->decode_active = true;
    
    // Every 5ms, submit a tiny command buffer if nothing else is queued
    ctx->keepalive_timer = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_TIMER, 0, 0, 
        dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0));
    
    dispatch_source_set_timer(ctx->keepalive_timer, 
        DISPATCH_TIME_NOW, 5 * NSEC_PER_MSEC, 1 * NSEC_PER_MSEC);
    
    dispatch_source_set_event_handler(ctx->keepalive_timer, ^{
        if (!ctx->decode_active) return;
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> noop = [queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [noop computeCommandEncoder];
        [enc endEncoding];
        [noop commit];
    });
    
    dispatch_resume(ctx->keepalive_timer);
}

void ggml_metal_stop_decode_session(ggml_metal_t ctx) {
    ctx->decode_active = false;
    if (ctx->keepalive_timer) {
        dispatch_source_cancel(ctx->keepalive_timer);
        ctx->keepalive_timer = nil;
    }
}
```

### Step 1.2: Replace waitUntilCompleted with Semaphore + CompletedHandler

File: `ggml/src/ggml-metal/ggml-metal-context.m`

The `waitUntilCompleted` call may have unnecessary overhead. Use a dispatch
semaphore instead (llama.cpp already does this for tensor copies):

```objc
void ggml_metal_synchronize(ggml_metal_t ctx) {
    if (ctx->cmd_buf_last) {
        // Fast path: check if already done
        if ([ctx->cmd_buf_last status] >= MTLCommandBufferStatusCompleted) {
            ctx->cmd_buf_last = nil;
            return;
        }
        
        // Use semaphore for faster wakeup than waitUntilCompleted
        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        [ctx->cmd_buf_last addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            dispatch_semaphore_signal(sem);
        }];
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        dispatch_release(sem);
        
        ctx->cmd_buf_last = nil;
    }
}
```

Note: llama.cpp's own code comments say this "should be faster, but don't seem to
make much difference" (line 1704 in ggml-metal-device.m). But worth verifying.


## Phase 2: The Real Fix -- MTLSharedEvent Pipeline (2-3 weeks)

### Step 2.1: Add Shared Events to Metal Context

File: `ggml/src/ggml-metal/ggml-metal-context.m`

```objc
struct ggml_metal {
    // ... existing fields ...
    
    // Double-buffered decode pipeline
    id<MTLSharedEvent> event_gpu_done;      // GPU signals when done
    id<MTLSharedEvent> event_input_ready;   // CPU signals when input is ready
    uint64_t event_counter;                 // monotonically increasing
    
    // Double-buffered command buffers for decode
    id<MTLCommandBuffer> decode_cb[2];
    int decode_cb_idx;  // alternates 0, 1
    
    // Shared memory for inter-step communication
    id<MTLBuffer> decode_token_buf;  // GPU writes argmax result here
    id<MTLBuffer> decode_input_buf;  // CPU writes next token here, GPU reads
};
```

### Step 2.2: Initialize Pipeline

```objc
void ggml_metal_init_decode_pipeline(ggml_metal_t ctx) {
    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
    
    ctx->event_gpu_done = [device newSharedEvent];
    ctx->event_input_ready = [device newSharedEvent];
    ctx->event_counter = 0;
    ctx->decode_cb_idx = 0;
    
    // Small buffers for token passing
    ctx->decode_token_buf = [device newBufferWithLength:sizeof(int32_t)
                                               options:MTLResourceStorageModeShared];
    ctx->decode_input_buf = [device newBufferWithLength:sizeof(int32_t)
                                               options:MTLResourceStorageModeShared];
}
```

### Step 2.3: Pipelined Graph Compute

```objc
enum ggml_status ggml_metal_graph_compute_pipelined(
    ggml_metal_t ctx, struct ggml_cgraph * gf, bool is_decode) {
    
    if (!is_decode) {
        // Fall back to standard path for prompt eval
        return ggml_metal_graph_compute(ctx, gf);
    }
    
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        int cb_idx = ctx->decode_cb_idx;
        uint64_t counter = ++ctx->event_counter;
        
        id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
        
        // Wait for CPU to signal that input token is ready
        [cmd_buf encodeWaitForEvent:ctx->event_input_ready value:counter];
        
        // Encode all compute operations
        ctx->gf = gf;
        ctx->n_nodes_0 = gf->n_nodes;
        ctx->n_nodes_1 = 0;
        
        // ... (encode ops same as current code) ...
        
        // Signal that GPU is done
        [cmd_buf encodeSignalEvent:ctx->event_gpu_done value:counter];
        
        // Commit immediately -- GPU will wait for event_input_ready
        [cmd_buf commit];
        
        ctx->decode_cb[cb_idx] = cmd_buf;
        ctx->decode_cb_idx = 1 - cb_idx;  // flip
    }
    
    return GGML_STATUS_SUCCESS;
}

// Called by CPU after writing the input token
void ggml_metal_signal_input_ready(ggml_metal_t ctx) {
    ctx->event_input_ready.signaledValue = ctx->event_counter;
}

// Wait for GPU to complete current decode step
void ggml_metal_wait_decode_done(ggml_metal_t ctx) {
    // Option A: Blocking wait
    while (ctx->event_gpu_done.signaledValue < ctx->event_counter) {
        // Spin-wait with yield (or use MTLSharedEventListener for async)
        usleep(10);  // 10us granularity
    }
    
    // Option B: Event listener (async, preferred)
    // Set up MTLSharedEventListener in init, notify via dispatch queue
}
```

### Step 2.4: Modified Decode Loop

File: `src/llama-context.cpp` (or wherever the decode loop lives)

```cpp
// Current flow:
for each token:
    build_graph()
    graph_compute()      // sync: encode + commit + waitUntilCompleted
    sample_token()       // CPU argmax
    
// New flow:
build_graph(token_0)
signal_input_ready()     // unblock GPU for token 0

for each token N:
    wait_decode_done()   // wait for token N GPU result
    result = read_output()
    
    build_graph(token_N+1)   // overlap: CPU builds while GPU idle time = 0
    signal_input_ready()     // unblock GPU for token N+1
    
    yield result             // stream token N to user
```

### Step 2.5: On-GPU Argmax (combine with Step 2.3)

File: `ggml/src/ggml-metal/ggml-metal.metal`

```metal
kernel void kernel_argmax_f32(
    device const float * src [[buffer(0)]],
    device int32_t * dst     [[buffer(1)]],
    constant int32_t & n     [[buffer(2)]],
    uint tpig [[thread_position_in_grid]],
    uint tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]) {
    
    // Each thread scans a chunk
    const int chunk = (n + 1023) / 1024;
    const int start = tpitg * chunk;
    const int end = min(start + chunk, n);
    
    float best_val = -INFINITY;
    int32_t best_idx = 0;
    
    for (int i = start; i < end; i++) {
        float v = src[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    
    // SIMD reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(best_val, offset);
        int32_t other_idx = simd_shuffle_down(best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }
    
    // Threadgroup reduction via shared memory
    threadgroup float sg_vals[32];
    threadgroup int32_t sg_idxs[32];
    
    if (tiisg == 0) {
        sg_vals[sgitg] = best_val;
        sg_idxs[sgitg] = best_idx;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tpitg == 0) {
        best_val = sg_vals[0];
        best_idx = sg_idxs[0];
        for (int i = 1; i < 32; i++) {
            if (sg_vals[i] > best_val) {
                best_val = sg_vals[i];
                best_idx = sg_idxs[i];
            }
        }
        dst[0] = best_idx;
    }
}
```

Chain this kernel at the end of the forward pass command buffer. The result
is written to `decode_token_buf` which is in shared memory -- CPU can read it
with a simple pointer dereference after `wait_decode_done()`.


## Phase 3: Optimization (1-2 weeks)

### Step 3.1: Investigate and Eliminate Graph Split #2

1. Run with `GGML_SCHED_DEBUG=2`
2. Identify which op causes the second split
3. Either move it to GPU or find a way to merge backends

### Step 3.2: Pre-fault Model Pages

Add at model load time:

```c
// In model loading code, after mmap:
void prefault_model(void * data, size_t size) {
    volatile char sum = 0;
    for (size_t i = 0; i < size; i += 16384) {  // 16KB pages on Apple Silicon
        sum += ((volatile char *)data)[i];
    }
    // This forces all pages into physical memory
    // Takes ~2-3 seconds for 10.4GB at SSD speed
}
```

This eliminates page faults during decode but doesn't help with TLB misses.

### Step 3.3: MTLSharedEventListener (Non-Blocking Wait)

Replace spin-wait in Step 2.4 with MTLSharedEventListener:

```objc
// In init:
ctx->event_listener = [[MTLSharedEventListener alloc] 
    initWithDispatchQueue:dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0)];

// In wait_decode_done:
dispatch_semaphore_t sem = dispatch_semaphore_create(0);
[ctx->event_gpu_done notifyListener:ctx->event_listener
                            atValue:ctx->event_counter
                              block:^(id<MTLSharedEvent> event, uint64_t value) {
    dispatch_semaphore_signal(sem);
}];
dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
```


## Expected Results

| Phase | Change | tok/s | ms/tok |
|-------|--------|-------|--------|
| Current | Baseline | 28 | 35.0 |
| Phase 0 | Measurement only | 28 | 35.0 |
| Phase 1 | Keep-alive + semaphore | 35-45 | 22-28 |
| Phase 2 | MTLSharedEvent pipeline | 100-200 | 5-10 |
| Phase 2+argmax | + on-GPU sampling | 125-250 | 4-8 |
| Phase 3 | + 1 split + prefault | 150-300 | 3-7 |

The 100 tok/s target is achievable with Phase 2 alone. Phases 2+3 could push
toward 200+ tok/s, approaching the memory bandwidth limit of Apple M4's GPU
(~100 GB/s reading ~400MB of active expert weights per token = 250 tok/s theoretical).
