#pragma once

// Expert-aware madvise prefetch for MoE models
//
// When expert weights are mmap'd from a GGUF file, consecutive tokens often
// activate similar experts (temporal locality). By reading which experts were
// activated after each token's forward pass, we can issue posix_madvise(MADV_WILLNEED)
// on the expert weight pages before the next token starts computing.
//
// On Apple Silicon with Metal (newBufferWithBytesNoCopy), the mmap pages ARE the
// GPU pages — so madvise directly warms the Metal working set.

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <atomic>

struct llama_model;

struct expert_prefetch_config {
    bool enabled       = false;  // master switch
    bool log_csv       = false;  // write CSV log for analysis
    std::string csv_path;        // path for CSV output
};

// Per-layer expert tensor info for prefetching
struct expert_tensor_info {
    void     * data;       // base pointer (into mmap)
    int64_t    n_expert;   // number of experts (ne[2])
    size_t     expert_stride; // bytes per expert (nb[2])
};

class llama_expert_prefetch {
public:
    llama_expert_prefetch();
    ~llama_expert_prefetch();

    // Initialize from model — extract expert tensor addresses for all MoE layers
    void init(const llama_model & model);

    // After graph_compute: extract activated expert indices from the graph,
    // issue madvise for those experts' weight pages, and optionally log data.
    // Returns the number of experts prefetched.
    int after_compute(
        struct ggml_cgraph * gf,
        struct ggml_backend_sched * sched,
        int token_id,           // sequential token number for logging
        double token_time_ms);  // time this token took to decode

    // Get/set config
    void set_config(const expert_prefetch_config & cfg);
    const expert_prefetch_config & get_config() const { return config; }

    // Statistics
    struct stats {
        int64_t total_tokens       = 0;
        int64_t total_prefetches   = 0;
        int64_t total_expert_overlap = 0; // count of experts same as previous token
        double  avg_overlap_ratio  = 0.0;
    };
    stats get_stats() const;

private:
    expert_prefetch_config config;

    // Expert tensor info per MoE layer (indexed by layer id)
    // gate_up_exps[layer_id] and down_exps[layer_id]
    std::vector<expert_tensor_info> gate_up_info;  // indexed by MoE layer
    std::vector<expert_tensor_info> down_info;
    std::vector<int> moe_layer_ids;  // which layer ids have MoE

    // Previous token's expert activations per layer (for overlap calculation)
    // prev_experts[moe_layer_idx] = vector of expert indices
    std::vector<std::vector<int32_t>> prev_experts;

    // Statistics
    std::atomic<int64_t> stat_tokens{0};
    std::atomic<int64_t> stat_prefetches{0};
    int64_t stat_overlap_sum = 0;
    int64_t stat_overlap_count = 0;

    // CSV log file
    FILE * csv_file = nullptr;

    void open_csv();
    void close_csv();

    // Issue madvise for a specific expert in a specific tensor
    void prefetch_expert(const expert_tensor_info & info, int32_t expert_idx);
};
