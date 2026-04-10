# KG-Optimization Engine: Architecture

## The Core Thesis

Current AI-driven optimization systems (AlphaEvolve, STARK, AutoKernel, K-Search) all share a fundamental limitation: they start from scratch for every new optimization problem. They don't accumulate structured knowledge about what works, why it works, and how optimizations relate to each other and to hardware.

A knowledge graph changes this by making optimization knowledge **persistent, queryable, composable, and transferable**. An agent traversing this graph doesn't blindly search — it reasons over structured relationships between hardware properties, optimization techniques, model architectures, and measured outcomes.

---

## Knowledge Graph Schema

### Node Types

```
HARDWARE
  ├── Chip(name, vendor, arch, compute_units, clock_mhz, tdp_watts)
  ├── MemoryLevel(name, type, size_bytes, bandwidth_gbps, latency_ns)
  ├── ComputeUnit(name, type, simd_width, ops_per_cycle, supported_dtypes)
  └── ISA(name, extensions[])

MODEL
  ├── Architecture(name, type, params, active_params)
  ├── Layer(name, type, input_shape, output_shape, flops)
  ├── Operator(name, compute_pattern, memory_pattern, arithmetic_intensity)
  └── Attention(type, heads, kv_heads, head_dim, context_length)

OPTIMIZATION
  ├── Technique(name, category, applicable_to[], incompatible_with[])
  ├── Quantization(name, bits, scheme, group_size, error_profile)
  ├── KernelImpl(name, target_hw, target_op, language, source_hash)
  ├── FusionPattern(ops_fused[], memory_saved, compute_overhead)
  └── ScheduleDecision(tiling[], parallelism[], vectorization[], memory_placement[])

MEASUREMENT
  ├── Benchmark(hw, model, config, throughput, latency, memory_peak, timestamp)
  ├── ProfileTrace(kernel, hw, cycles, cache_hits, cache_misses, occupancy, bandwidth_util)
  └── BottleneckAnalysis(kernel, hw, bottleneck_type, utilization_pct, headroom_pct)

COMPOSITION
  ├── Pipeline(stages[], total_latency, total_memory)
  ├── OptimizationPlan(techniques[], expected_gain, measured_gain, validated)
  └── Discovery(source_plan, novel_combination, improvement_pct, mechanism_hypothesis)
```

### Edge Types

```
HARDWARE TOPOLOGY
  Chip --HAS_MEMORY--> MemoryLevel (with: level, shared_by)
  Chip --HAS_COMPUTE--> ComputeUnit (with: count)
  MemoryLevel --FEEDS--> MemoryLevel (with: bandwidth, latency)
  MemoryLevel --FEEDS--> ComputeUnit

MODEL STRUCTURE
  Architecture --CONTAINS--> Layer (with: position, count)
  Layer --USES--> Operator (with: count_per_forward)
  Layer --HAS_ATTENTION--> Attention
  Architecture --INSTANCE_OF--> Architecture (e.g., Gemma4 --INSTANCE_OF--> MoE)

OPTIMIZATION APPLICABILITY
  Technique --APPLIES_TO--> Operator (with: expected_speedup, conditions[])
  Technique --REQUIRES--> ComputeUnit (with: min_capability)
  Technique --CONFLICTS_WITH--> Technique
  Technique --COMPOSES_WITH--> Technique (with: combined_effect)
  Quantization --TARGETS--> Layer (with: sensitivity, accuracy_impact)
  KernelImpl --IMPLEMENTS--> Operator (with: for_hardware)
  FusionPattern --FUSES--> Operator[]

MEASUREMENT LINKS
  Benchmark --MEASURES--> Pipeline
  Benchmark --ON_HARDWARE--> Chip
  ProfileTrace --PROFILES--> KernelImpl
  BottleneckAnalysis --IDENTIFIES--> Operator (with: bottleneck_type)

DISCOVERY PROVENANCE
  Discovery --DERIVED_FROM--> OptimizationPlan[]
  Discovery --VALIDATED_BY--> Benchmark
  Discovery --TRANSFERS_TO--> Chip[] (with: confidence)
  OptimizationPlan --IMPROVES--> Pipeline (with: measured_delta)
```

### Key Properties on Nodes

Every node carries:
- `created_at`, `updated_at` timestamps
- `confidence` score (0-1, based on measurement count and recency)
- `provenance` (paper, experiment, human expert, agent-discovered)
- `embedding` (vector embedding for similarity search)

---

## Agent Architecture

### Query/Traversal Pattern

The agent operates in four modes:

#### 1. Bottleneck Discovery
```
Given: Model M running on Hardware H
1. Query: MATCH (m:Architecture {name: M})-[:CONTAINS]->(l:Layer)-[:USES]->(op:Operator)
          MATCH (b:BottleneckAnalysis)-[:IDENTIFIES]->(op) WHERE b.hw = H
          RETURN op ORDER BY b.headroom_pct DESC
2. For each bottleneck operator, traverse optimization edges:
          MATCH (t:Technique)-[:APPLIES_TO]->(op) WHERE NOT (t)-[:CONFLICTS_WITH]->(:Technique IN current_plan)
          RETURN t ORDER BY t.expected_speedup * feasibility_score DESC
```

#### 2. Novel Combination Search
```
Given: A set of known-good techniques T1, T2, T3
1. Find techniques that COMPOSE_WITH each of T1, T2, T3 but haven't been measured together
2. Check CONFLICTS_WITH edges to prune impossible combinations
3. Estimate combined effect using composition rules on COMPOSES_WITH edges
4. Rank by (estimated_gain * confidence) / validation_cost
5. Generate: OptimizationPlan with the novel combination
```

#### 3. Cross-Hardware Transfer
```
Given: Discovery D validated on Hardware H1, Target Hardware H2
1. Compare H1 and H2 via memory hierarchy, compute unit, ISA similarities
2. For each technique in D, check REQUIRES edges against H2 capabilities
3. Find TRANSFERS_TO edges from similar past discoveries
4. Adjust expected gains based on hardware delta (e.g., bandwidth ratio)
5. Output: adapted OptimizationPlan for H2 with confidence score
```

#### 4. Evolutionary Refinement (AlphaEvolve-style)
```
Given: KernelImpl K with ProfileTrace P showing headroom
1. Traverse KG for similar kernels with better profiles
2. Diff the implementations: what techniques do better kernels use?
3. Generate mutation proposals informed by KG relationships
4. Evaluate mutations, update KG with results
5. If improvement: create Discovery node with provenance chain
```

### Agent Loop

```
while True:
    # 1. Observe: profile current system
    traces = profile(current_pipeline)
    update_kg(traces)

    # 2. Analyze: find bottlenecks via graph queries
    bottlenecks = query_bottlenecks(kg, model, hardware)

    # 3. Plan: traverse optimization graph for novel combinations
    candidates = search_novel_combinations(kg, bottlenecks)

    # 4. Generate: produce concrete implementations
    for plan in candidates:
        kernel = generate_kernel(plan, kg_context)

        # 5. Validate: correctness + performance
        result = validate(kernel, correctness_tests, perf_benchmark)

        # 6. Learn: update knowledge graph
        if result.correct and result.faster:
            create_discovery_node(kg, plan, result)
        else:
            record_negative_result(kg, plan, result)  # failures are knowledge too
```

---

## Data Sources to Populate the KG

### Hardware Data (Automated)
- **Apple Silicon**: `sysctl` for CPU/GPU specs, Metal Feature Set tables, IOKit GPU properties
- **NVIDIA**: CUDA device properties API, `nvidia-smi`, architecture whitepapers
- **AMD**: ROCm device info, RDNA/CDNA architecture docs
- **General**: Chip-specific memory bandwidth benchmarks, cache hierarchy measurements

### Model Data (Automated)
- **GGUF metadata**: Layer counts, tensor shapes, quantization info from model files
- **ONNX/SafeTensors**: Operator graphs, shapes, dtypes
- **Profiling**: Per-layer timing from inference runs, memory allocation traces

### Optimization Knowledge (Semi-automated)
- **Papers**: Parse optimization papers for technique descriptions, results, hardware targets
- **Compiler IRs**: Extract optimization passes from TVM, MLIR, XLA schedules
- **Kernel repositories**: CUTLASS, ThunderKittens, Triton kernels — extract patterns
- **Benchmark databases**: MLPerf, KernelBench, TritonGym results

### Discovered Knowledge (Agent-generated)
- Every optimization attempt (success or failure) becomes a graph edge
- Profiling traces from generated kernels populate measurement nodes
- Novel combinations that produce improvements become Discovery nodes

---

## How the Agent Generates and Validates Novel Optimization Proposals

### Generation: Graph-Guided Mutation

1. **Structural proposals**: Traverse COMPOSES_WITH edges to find untested technique combinations
2. **Analogical proposals**: Find similar (operator, hardware) pairs where an optimization worked, transfer to current target
3. **Interpolation proposals**: If technique A gives 2x on operator O and technique B gives 1.5x on O, try A+B
4. **Extrapolation proposals**: If tiling size 64 is optimal for 4K context and 128 for 8K, predict optimal for 32K

### Validation Pipeline (adapted from AutoKernel)

```
Stage 1: Smoke Test     — runs on small input, checks output shape and dtype
Stage 2: Correctness    — compare output against reference implementation (rtol=1e-3)
Stage 3: Shape Sweep    — test across representative input shapes
Stage 4: Numerical      — check for NaN/Inf, test edge cases (zero, max, denormals)
Stage 5: Performance    — benchmark latency, throughput, memory on target hardware
Stage 6: Regression     — ensure no regression on previously passing tests
Stage 7: Stability      — run 1000x, check determinism and variance
```

### Feedback Loop

- **Positive results**: New Discovery node + update confidence scores on involved technique nodes
- **Negative results**: Record failure mode, update CONFLICTS_WITH edges if techniques are incompatible
- **Partial results**: If correct but slower, record as data point for future analysis
- **Surprising results**: Flag for human review if improvement exceeds 3x expected

---

## MVP Scope: The Smallest Useful Version

### MVP Target: Metal Kernel Optimization for MoE Inference on Apple Silicon

Why this scope:
1. We already have the inference pipeline (LocalCode + llama.cpp fork)
2. Apple Silicon Metal is underserved by optimization tools
3. MoE models have huge optimization potential (K-Search showed 14.3x on MoE kernels)
4. Small hardware surface area (M-series chips share architecture)
5. We have ground truth benchmarks (27 tok/s baseline)

### MVP Components

**KG (Phase 1 — Static)**
- 5 hardware nodes: M4 memory hierarchy (unified memory, L1, L2, SLC, DRAM)
- 10 operator nodes: attention, MoE routing, expert FFN, RMSNorm, embedding, etc.
- 20 technique nodes: tiling strategies, fusion patterns, quantization variants
- 50 measurement nodes: baseline profiling of current llama.cpp inference
- Stored in: NetworkX graph (in-memory) + JSON serialization

**Agent (Phase 1 — Simple)**
- Single agent with 3 tools: query_kg, generate_kernel, validate_kernel
- Uses Claude/GPT-4 as the LLM backbone
- Bottleneck-first strategy: always optimize the slowest kernel first
- Evolutionary refinement: mutate best-known kernel, evaluate, update KG

**Evaluator (Phase 1 — Reuse existing)**
- Compile Metal kernels via Xcode toolchain
- Benchmark via Metal performance counters
- Correctness check against reference llama.cpp output

### MVP Success Criteria

- Discovers at least one optimization that improves tok/s by >5%
- Accumulates >100 measurement nodes in the KG over a week of running
- Demonstrates that KG-guided search converges faster than random mutation
- Produces a human-readable optimization report from the KG

### MVP Timeline: 4 weeks

- Week 1: Build KG schema + populate with Metal/M4 hardware data and baseline measurements
- Week 2: Build agent loop (query KG -> generate kernel -> validate -> update KG)
- Week 3: Run agent on MoE expert dispatch + attention kernels, accumulate data
- Week 4: Analyze results, compare KG-guided vs random search, write up findings

---

## Why This Is Defensible as a Product/Company

### The Moat: Accumulated Optimization Knowledge

1. **Network effects**: Every user running the agent generates optimization data. More data = better KG = better optimizations for everyone. This compounds.

2. **Hardware coverage**: The KG accumulates knowledge across hardware targets. As new chips launch (M5, next-gen NVIDIA, custom AI accelerators), the KG transfers existing knowledge and adapts. Competitors starting from scratch can't match this.

3. **Cross-domain transfer**: An optimization discovered for attention kernels on M4 might transfer to convolutions on NVIDIA. The KG makes these connections explicit and searchable. No human could track all these relationships.

4. **Negative knowledge**: Failed optimizations are as valuable as successes. The KG encodes what doesn't work and why, preventing the agent (and users) from repeating mistakes. This is knowledge that papers don't publish.

### Product Angles

**For chip companies**: "Here's a system that automatically discovers optimal kernels for your new hardware, using knowledge from all prior hardware generations." Intel, Qualcomm, startups designing custom AI accelerators all need this.

**For ML framework teams**: "Plug this into your compilation pipeline. It generates optimized kernels for any operator on any supported hardware, improving over time." Replaces hand-tuned kernel libraries.

**For inference providers**: "Run this continuously against your production workloads. It finds and validates optimizations automatically, reducing cost-per-token." The 2.5-5x speedups demonstrated by existing systems translate directly to cost savings.

**For hardware-software co-design**: "Use the KG to explore which hardware changes would unlock the most software optimizations." Chip architects get data-driven guidance.

### Competitive Landscape

| Player | Approach | Limitation |
|--------|----------|------------|
| AlphaEvolve | Evolutionary + LLM | Google-internal, starts fresh each time |
| TVM/MLIR | Compiler autotuning | No accumulated knowledge, no LLM guidance |
| STARK/AutoKernel | Multi-agent kernel optimization | No persistent KG, single-problem focus |
| ThunderKittens | DSL for GPU kernels | Manual optimization, no automation |
| **KG-Optimization Engine** | **KG + Agent + Evolutionary** | **Accumulates knowledge, transfers across hardware/models** |

The key differentiator: everyone else optimizes one problem at a time. We build a knowledge base that makes every subsequent optimization faster and better. This is the difference between a tool and a platform.
