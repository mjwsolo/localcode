#include "llama-expert-prefetch.h"
#include "llama-model.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <set>
#include <thread>

#ifdef _POSIX_MAPPED_FILES
#include <sys/mman.h>
#endif

#ifndef _WIN32
#include <sys/resource.h>  // for getrusage page fault counting
#endif

llama_expert_prefetch::llama_expert_prefetch() = default;

llama_expert_prefetch::~llama_expert_prefetch() {
    close_csv();
}

void llama_expert_prefetch::init(const llama_model & model) {
    const auto & hparams = model.hparams;
    const uint32_t n_layer = hparams.n_layer;

    gate_up_info.clear();
    down_info.clear();
    moe_layer_ids.clear();
    prev_experts.clear();

    for (uint32_t il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];

        // Check if this layer has MoE expert tensors
        if (layer.ffn_gate_up_exps == nullptr && layer.ffn_down_exps == nullptr) {
            continue;
        }

        moe_layer_ids.push_back(il);

        expert_tensor_info gu_info = {};
        expert_tensor_info dn_info = {};

        if (layer.ffn_gate_up_exps) {
            gu_info.data          = layer.ffn_gate_up_exps->data;
            gu_info.n_expert      = layer.ffn_gate_up_exps->ne[2];
            gu_info.expert_stride = layer.ffn_gate_up_exps->nb[2];
        }
        if (layer.ffn_down_exps) {
            dn_info.data          = layer.ffn_down_exps->data;
            dn_info.n_expert      = layer.ffn_down_exps->ne[2];
            dn_info.expert_stride = layer.ffn_down_exps->nb[2];
        }

        gate_up_info.push_back(gu_info);
        down_info.push_back(dn_info);
    }

    // Initialize previous expert tracking
    prev_experts.resize(moe_layer_ids.size());

    LLAMA_LOG_INFO("%s: initialized expert prefetch for %zu MoE layers\n",
                   __func__, moe_layer_ids.size());

    if (!moe_layer_ids.empty()) {
        const auto & first_gu = gate_up_info[0];
        const auto & first_dn = down_info[0];
        LLAMA_LOG_INFO("%s: gate_up_exps: %d experts, %.1f MiB/expert | down_exps: %d experts, %.1f MiB/expert\n",
                       __func__,
                       (int)first_gu.n_expert,
                       first_gu.expert_stride / (1024.0 * 1024.0),
                       (int)first_dn.n_expert,
                       first_dn.expert_stride / (1024.0 * 1024.0));
    }
}

void llama_expert_prefetch::set_config(const expert_prefetch_config & cfg) {
    close_csv();
    config = cfg;
    if (config.log_csv && !config.csv_path.empty()) {
        open_csv();
    }
}

void llama_expert_prefetch::open_csv() {
    csv_file = fopen(config.csv_path.c_str(), "w");
    if (csv_file) {
        fprintf(csv_file, "token_id,time_ms,toks_per_sec,layer_id,experts,overlap_count,overlap_ratio,page_faults\n");
        fflush(csv_file);
    }
}

void llama_expert_prefetch::close_csv() {
    if (csv_file) {
        fclose(csv_file);
        csv_file = nullptr;
    }
}

static int64_t get_page_faults() {
#ifndef _WIN32
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        // ru_majflt = major page faults (required I/O)
        // ru_minflt = minor page faults (no I/O, but still a fault)
        return ru.ru_majflt;
    }
#endif
    return -1;
}

void llama_expert_prefetch::prefetch_expert(const expert_tensor_info & info, int32_t expert_idx) {
    if (!info.data || expert_idx < 0 || expert_idx >= info.n_expert) {
        return;
    }

#ifdef _POSIX_MAPPED_FILES
    void * expert_ptr = (char *)info.data + (size_t)expert_idx * info.expert_stride;
    size_t expert_size = info.expert_stride;

    // Align to page boundary (use 16KB for Apple Silicon)
    const size_t page_size = 16384;
    uintptr_t addr = (uintptr_t)expert_ptr;
    uintptr_t page_start = addr & ~(uintptr_t)(page_size - 1);
    size_t aligned_size = expert_size + (addr - page_start);
    // Round up to page boundary
    aligned_size = (aligned_size + page_size - 1) & ~(page_size - 1);

    posix_madvise((void *)page_start, aligned_size, POSIX_MADV_WILLNEED);
#else
    (void)info;
    (void)expert_idx;
#endif
}

int llama_expert_prefetch::after_compute(
        struct ggml_cgraph * gf,
        struct ggml_backend_sched * sched,
        int token_id,
        double token_time_ms) {

    if (moe_layer_ids.empty()) {
        return 0;
    }

    int total_prefetched = 0;
    int64_t pf_before = config.log_csv ? get_page_faults() : 0;

    // Build a map from layer_id -> moe_idx for quick lookup
    // (could cache this, but it's tiny)
    std::vector<std::pair<int, ggml_tensor *>> found_topk; // (moe_idx, tensor)

    // Single pass over graph nodes to find all ffn_moe_topk-* tensors
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes && found_topk.size() < moe_layer_ids.size(); i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!node || node->name[0] == '\0') continue;

        // Check if name matches "ffn_moe_topk-{N}"
        if (strncmp(node->name, "ffn_moe_topk-", 13) != 0) continue;

        int layer_id = atoi(node->name + 13);

        // Find corresponding moe_idx
        for (size_t idx = 0; idx < moe_layer_ids.size(); idx++) {
            if (moe_layer_ids[idx] == layer_id) {
                found_topk.emplace_back((int)idx, node);
                break;
            }
        }
    }

    for (const auto & entry : found_topk) {
        const size_t moe_idx = (size_t)entry.first;
        ggml_tensor * topk_tensor = entry.second;
        int layer_id = moe_layer_ids[moe_idx];

        // topk_tensor shape: [n_expert_used, n_tokens]
        // For single-token decode, n_tokens = 1
        // Type: I32
        const int64_t n_expert_used = topk_tensor->ne[0];
        const int64_t n_tokens = topk_tensor->ne[1];

        if (n_tokens != 1) {
            // Only do prefetch for single-token decode (not prompt eval)
            continue;
        }

        // Read expert indices from the tensor (may be on GPU)
        std::vector<int32_t> expert_indices(n_expert_used);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, topk_tensor);
        if (backend) {
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(topk_tensor, expert_indices.data(), 0, n_expert_used * sizeof(int32_t));
        } else {
            // Tensor might be on CPU
            if (topk_tensor->data) {
                memcpy(expert_indices.data(), topk_tensor->data, n_expert_used * sizeof(int32_t));
            } else {
                continue;
            }
        }

        // Calculate overlap with previous token's experts for this layer
        int overlap_count = 0;
        if (!prev_experts[moe_idx].empty()) {
            std::set<int32_t> prev_set(prev_experts[moe_idx].begin(), prev_experts[moe_idx].end());
            for (int32_t eidx : expert_indices) {
                if (prev_set.count(eidx)) {
                    overlap_count++;
                }
            }
        }

        float overlap_ratio = prev_experts[moe_idx].empty() ? 0.0f :
            (float)overlap_count / (float)n_expert_used;

        // Issue prefetch for these experts (betting on temporal locality)
        if (config.enabled) {
            for (int32_t eidx : expert_indices) {
                if (moe_idx < gate_up_info.size()) {
                    prefetch_expert(gate_up_info[moe_idx], eidx);
                    total_prefetched++;
                }
                if (moe_idx < down_info.size()) {
                    prefetch_expert(down_info[moe_idx], eidx);
                    total_prefetched++;
                }
            }
        }

        // Log CSV
        if (csv_file) {
            int64_t pf_now = get_page_faults();
            double tps = token_time_ms > 0 ? 1000.0 / token_time_ms : 0.0;

            // Format expert list
            std::string expert_list;
            for (size_t i = 0; i < expert_indices.size(); i++) {
                if (i > 0) expert_list += ";";
                expert_list += std::to_string(expert_indices[i]);
            }

            fprintf(csv_file, "%d,%.3f,%.1f,%d,%s,%d,%.3f,%lld\n",
                    token_id, token_time_ms, tps,
                    layer_id, expert_list.c_str(),
                    overlap_count, overlap_ratio,
                    (long long)(pf_now - pf_before));
            fflush(csv_file);
        }

        // Update tracking
        prev_experts[moe_idx] = expert_indices;

        // Update stats
        if (!prev_experts[moe_idx].empty()) {
            stat_overlap_sum += overlap_count;
            stat_overlap_count++;
        }
    }

    stat_tokens++;
    stat_prefetches += total_prefetched;

    return total_prefetched;
}

llama_expert_prefetch::stats llama_expert_prefetch::get_stats() const {
    stats s;
    s.total_tokens = stat_tokens.load();
    s.total_prefetches = stat_prefetches.load();
    s.total_expert_overlap = stat_overlap_sum;
    s.avg_overlap_ratio = stat_overlap_count > 0 ?
        (double)stat_overlap_sum / (double)stat_overlap_count : 0.0;
    return s;
}
