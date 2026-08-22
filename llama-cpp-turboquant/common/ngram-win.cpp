#include "ngram-win.h"

#include <algorithm>
#include <cstring>

//
// common_ngram_win
//

common_ngram_win::common_ngram_win(
    uint16_t window_size,
    const std::vector<uint16_t> & levels,
    size_t table_size)
    : window_size(window_size)
    , levels(levels)
    , table_size(table_size)
{
    // Sort levels descending (longest match first)
    std::sort(this->levels.begin(), this->levels.end(), std::greater<uint16_t>());

    // Allocate tables
    tables.resize(this->levels.size());
    for (auto & table : tables) {
        table.resize(table_size);
    }

    // Allocate ring buffer
    ring.resize(window_size, EMPTY);

    reset();
}

void common_ngram_win::push(entry_t token) {
    ring[ring_pos] = token;
    ring_pos = (ring_pos + 1) % window_size;
    if (ring_count < window_size) {
        ring_count++;
    }
    tokens_since_rebuild++;

    // Auto-rebuild periodically
    if (tokens_since_rebuild >= rebuild_interval) {
        rebuild();
    }
}

size_t common_ngram_win::hash_n(const entry_t * tokens, uint16_t n) const {
    size_t h = 0;
    for (uint16_t i = 0; i < n; ++i) {
        h = h * 6364136223846793005ULL + (size_t)tokens[i];
    }
    return h % table_size;
}

void common_ngram_win::rebuild() {
    // Clear all tables
    for (auto & table : tables) {
        for (auto & entry : table) {
            entry.token = EMPTY;
            entry.count = 0;
        }
    }

    tokens_since_rebuild = 0;

    if (ring_count < 2) {
        return;
    }

    // Linearize the ring buffer into a contiguous array
    // This is simpler and the rebuild only happens every ~64 tokens
    std::vector<entry_t> linear(ring_count);
    if (ring_count < window_size) {
        // Ring hasn't wrapped yet
        std::memcpy(linear.data(), ring.data(), ring_count * sizeof(entry_t));
    } else {
        // Ring has wrapped: [ring_pos..end] + [0..ring_pos)
        size_t tail = window_size - ring_pos;
        std::memcpy(linear.data(), ring.data() + ring_pos, tail * sizeof(entry_t));
        std::memcpy(linear.data() + tail, ring.data(), ring_pos * sizeof(entry_t));
    }

    // Build n-grams at each level
    for (size_t li = 0; li < levels.size(); ++li) {
        uint16_t n = levels[li];
        auto & table = tables[li];

        if ((size_t)n >= ring_count) {
            continue;
        }

        for (size_t i = 0; i + n < ring_count; ++i) {
            size_t idx = hash_n(linear.data() + i, n);
            entry_t next = linear[i + n];

            if (table[idx].token == EMPTY) {
                table[idx].token = next;
                table[idx].count = 1;
            } else if (table[idx].token == next) {
                table[idx].count++;
            } else {
                // Hash collision: replace if new pattern is more frequent
                // (simple frequency tracking: decrement old, set new if 0)
                if (table[idx].count <= 1) {
                    table[idx].token = next;
                    table[idx].count = 1;
                } else {
                    table[idx].count--;
                }
            }
        }
    }
}

common_ngram_win::entry_t common_ngram_win::predict_one(
        const entry_t * context, size_t context_len) const {
    // Try each level from longest to shortest
    for (size_t li = 0; li < levels.size(); ++li) {
        uint16_t n = levels[li];

        if (context_len < (size_t)n) {
            continue;
        }

        const entry_t * key = context + context_len - n;
        size_t idx = hash_n(key, n);

        const auto & entry = tables[li][idx];
        if (entry.token != EMPTY) {
            return entry.token;
        }
    }

    return EMPTY;
}

int common_ngram_win::predict_chain(
        const entry_t * context, size_t context_len,
        entry_t * draft, int k) const {

    // Build a temporary extended context for chaining
    size_t max_n = levels.empty() ? 0 : levels[0];
    size_t buf_size = max_n + k;

    // Use stack allocation for small buffers
    entry_t stack_buf[64];
    entry_t * buf = (buf_size <= 64) ? stack_buf : new entry_t[buf_size];

    // Copy the tail of context into buf
    size_t copy_start = (context_len > max_n) ? (context_len - max_n) : 0;
    size_t copy_len = context_len - copy_start;
    std::memcpy(buf, context + copy_start, copy_len * sizeof(entry_t));

    int n_predicted = 0;
    for (int i = 0; i < k; ++i) {
        entry_t pred = predict_one(buf, copy_len + i);
        if (pred == EMPTY) {
            break;
        }
        draft[i] = pred;
        buf[copy_len + i] = pred;
        n_predicted++;
    }

    if (buf != stack_buf) {
        delete[] buf;
    }

    return n_predicted;
}

void common_ngram_win::reset() {
    for (auto & table : tables) {
        for (auto & entry : table) {
            entry.token = EMPTY;
            entry.count = 0;
        }
    }

    std::fill(ring.begin(), ring.end(), EMPTY);
    ring_pos = 0;
    ring_count = 0;
    tokens_since_rebuild = 0;
}

size_t common_ngram_win::size_bytes() const {
    size_t bytes = ring.size() * sizeof(entry_t); // ring buffer
    for (const auto & table : tables) {
        bytes += table.size() * sizeof(table_entry); // hash tables
    }
    return bytes;
}

size_t common_ngram_win::get_used() const {
    size_t used = 0;
    for (const auto & table : tables) {
        for (const auto & entry : table) {
            if (entry.token != EMPTY) {
                used++;
            }
        }
    }
    return used;
}
