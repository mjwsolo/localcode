#!/usr/bin/env python3
"""
Analyze expert activation patterns from llama-server expert prefetch logs.

Usage:
    # First, run llama-server with expert logging:
    LLAMA_EXPERT_LOG=/tmp/expert_log.csv llama-server ...

    # Then analyze:
    python analyze_experts.py /tmp/expert_log.csv
"""

import sys
import csv
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path


def load_csv(path):
    """Load expert activation CSV log."""
    records = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'token_id': int(row['token_id']),
                'time_ms': float(row['time_ms']),
                'toks_per_sec': float(row['toks_per_sec']),
                'layer_id': int(row['layer_id']),
                'experts': [int(x) for x in row['experts'].split(';') if x],
                'overlap_count': int(row['overlap_count']),
                'overlap_ratio': float(row['overlap_ratio']),
                'page_faults': int(row['page_faults']),
            })
    return records


def analyze(records):
    """Analyze expert activation patterns."""
    if not records:
        print("No records found.")
        return

    # Group by token
    tokens = defaultdict(list)
    for r in records:
        tokens[r['token_id']].append(r)

    n_tokens = len(tokens)
    print(f"\n{'='*70}")
    print(f"Expert Activation Analysis: {n_tokens} tokens, {len(records)} layer-records")
    print(f"{'='*70}")

    # Per-token timing
    token_times = {}
    for tid, layers in tokens.items():
        token_times[tid] = layers[0]['time_ms']  # all layers same token time

    times = np.array(list(token_times.values()))
    print(f"\n--- Per-Token Decode Latency ---")
    print(f"  Mean:   {np.mean(times):7.1f} ms ({1000/np.mean(times):5.1f} tok/s)")
    print(f"  Median: {np.median(times):7.1f} ms ({1000/np.median(times):5.1f} tok/s)")
    print(f"  Min:    {np.min(times):7.1f} ms ({1000/np.min(times):5.1f} tok/s)")
    print(f"  Max:    {np.max(times):7.1f} ms ({1000/np.max(times):5.1f} tok/s)")
    print(f"  Stddev: {np.std(times):7.1f} ms")
    print(f"  CV:     {np.std(times)/np.mean(times)*100:.1f}%")

    # Expert overlap vs timing correlation
    print(f"\n--- Expert Overlap vs Latency Correlation ---")

    # Compute average overlap per token across all layers
    token_overlaps = {}
    for tid, layers in tokens.items():
        overlaps = [l['overlap_ratio'] for l in layers]
        token_overlaps[tid] = np.mean(overlaps) if overlaps else 0

    # Need at least token 1 (token 0 has no prev)
    valid_tids = sorted([t for t in token_overlaps if t > 0 and t in token_times])

    if len(valid_tids) > 2:
        x_overlap = np.array([token_overlaps[t] for t in valid_tids])
        y_time = np.array([token_times[t] for t in valid_tids])

        corr = np.corrcoef(x_overlap, y_time)[0, 1]
        print(f"  Correlation (overlap vs time): {corr:.3f}")
        print(f"  {'NEGATIVE = overlap predicts speed (GOOD!)' if corr < -0.2 else 'Weak or no correlation'}")

        # Bucket analysis
        low_overlap = y_time[x_overlap < 0.3]
        mid_overlap = y_time[(x_overlap >= 0.3) & (x_overlap < 0.6)]
        high_overlap = y_time[x_overlap >= 0.6]

        print(f"\n  Low overlap  (<30%): n={len(low_overlap):3d}, "
              f"mean={np.mean(low_overlap):6.1f}ms" if len(low_overlap) > 0 else "")
        print(f"  Mid overlap (30-60%): n={len(mid_overlap):3d}, "
              f"mean={np.mean(mid_overlap):6.1f}ms" if len(mid_overlap) > 0 else "")
        print(f"  High overlap (>60%): n={len(high_overlap):3d}, "
              f"mean={np.mean(high_overlap):6.1f}ms" if len(high_overlap) > 0 else "")

        if len(low_overlap) > 0 and len(high_overlap) > 0:
            speedup = np.mean(low_overlap) / np.mean(high_overlap)
            print(f"\n  Speed ratio (low/high overlap): {speedup:.2f}x")
            if speedup > 1.5:
                print(f"  >>> BREAKTHROUGH: High expert overlap = {speedup:.1f}x faster! <<<")
    else:
        print("  Not enough valid tokens for correlation analysis")

    # Page fault analysis
    print(f"\n--- Page Fault Analysis ---")
    pf_values = [r['page_faults'] for r in records if r['page_faults'] >= 0]
    if pf_values:
        pf = np.array(pf_values)
        print(f"  Mean page faults/layer-compute: {np.mean(pf):.1f}")
        print(f"  Max:  {np.max(pf)}")
        print(f"  Total: {np.sum(pf)}")

        # Page faults vs timing
        pf_per_token = {}
        for tid, layers in tokens.items():
            pf_per_token[tid] = sum(l['page_faults'] for l in layers)

        valid_pf = sorted([t for t in pf_per_token if t in token_times])
        if len(valid_pf) > 2:
            x_pf = np.array([pf_per_token[t] for t in valid_pf])
            y_time_pf = np.array([token_times[t] for t in valid_pf])
            corr_pf = np.corrcoef(x_pf, y_time_pf)[0, 1]
            print(f"  Correlation (page faults vs time): {corr_pf:.3f}")
    else:
        print("  No page fault data available")

    # Expert frequency analysis
    print(f"\n--- Expert Activation Frequency ---")
    expert_counter = Counter()
    for r in records:
        for e in r['experts']:
            expert_counter[e] += 1

    total_activations = sum(expert_counter.values())
    n_unique = len(expert_counter)
    print(f"  Unique experts activated: {n_unique}")
    print(f"  Total activations: {total_activations}")

    # Top 20 most common experts
    print(f"\n  Top 20 most activated experts:")
    for expert_id, count in expert_counter.most_common(20):
        pct = count / total_activations * 100
        print(f"    Expert {expert_id:3d}: {count:5d} ({pct:4.1f}%)")

    # Per-layer expert diversity
    print(f"\n--- Per-Layer Expert Diversity ---")
    layer_experts = defaultdict(set)
    for r in records:
        for e in r['experts']:
            layer_experts[r['layer_id']].add(e)

    for lid in sorted(layer_experts.keys())[:10]:
        print(f"  Layer {lid:2d}: {len(layer_experts[lid]):3d} unique experts used")
    if len(layer_experts) > 10:
        print(f"  ... ({len(layer_experts)} total MoE layers)")

    # Temporal expert pattern
    print(f"\n--- Temporal Expert Pattern (first 20 tokens, layer 0) ---")
    layer0_records = [r for r in records if r['layer_id'] == min(r['layer_id'] for r in records)]
    for r in sorted(layer0_records, key=lambda x: x['token_id'])[:20]:
        experts_str = ','.join(f'{e:3d}' for e in sorted(r['experts']))
        overlap_str = f"overlap={r['overlap_ratio']:.2f}" if r['token_id'] > 0 else "first"
        print(f"  Token {r['token_id']:3d} [{r['time_ms']:6.1f}ms]: [{experts_str}] {overlap_str}")

    # Summary and recommendation
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    if len(valid_tids) > 2:
        if corr < -0.3:
            print(f"STRONG CORRELATION FOUND: Expert overlap strongly predicts speed.")
            print(f"Expert prefetching (madvise) is highly likely to improve performance.")
            print(f"Expected improvement: up to {speedup:.1f}x on warm tokens.")
        elif corr < -0.1:
            print(f"MODERATE CORRELATION: Some relationship between overlap and speed.")
            print(f"Expert prefetching may help but gains may be modest.")
        else:
            print(f"WEAK/NO CORRELATION: Expert overlap does not strongly predict speed.")
            print(f"The latency variance may be caused by other factors (GPU scheduling, etc.)")

    cv = np.std(times)/np.mean(times)*100
    print(f"\nLatency variance (CV={cv:.0f}%) is {'HIGH' if cv > 30 else 'MODERATE' if cv > 15 else 'LOW'}.")
    print(f"Fastest token: {np.min(times):.1f}ms = {1000/np.min(times):.0f} tok/s")
    print(f"If all tokens were this fast: {1000/np.min(times):.0f} tok/s (vs current {1000/np.mean(times):.0f} tok/s)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <expert_log.csv>")
        sys.exit(1)

    records = load_csv(sys.argv[1])
    analyze(records)
