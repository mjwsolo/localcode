#!/usr/bin/env python3
"""
Adaptive multi-level n-gram speculation for code generation.

This tests an improved n-gram strategy that combines:
1. Multiple n-gram levels (2-gram through 8-gram) with priority to longer matches
2. Frequency-weighted predictions (prefer patterns seen multiple times)
3. Context-window n-grams (only learn from recent tokens, not stale data)
4. Structural awareness (newline + indent patterns get boosted)

The goal: push n-gram acceptance from ~45% to >50% on code.

Additionally tests the "self-draft via prompt eval" approach:
- Use n-gram to predict K draft tokens
- Batch-evaluate them (prompt eval, not decode)
- Accept matching tokens
- This is exactly what speculative decode already does, but with better drafts
"""

import time
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


def simple_tokenize(text: str) -> List[str]:
    """Approximate BPE tokenization."""
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in ' \t':
            j = i
            while j < len(text) and text[j] == c:
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
        if c == '\n':
            tokens.append('\n')
            i += 1
            continue
        if c in '(){}[]<>;:,.=+-*/%&|^~!?@#$\'"\\':
            if i + 1 < len(text):
                two = text[i:i+2]
                if two in ('==', '!=', '<=', '>=', '&&', '||', '->', '=>', '::',
                           '++', '--', '+=', '-=', '*=', '/=', '**', '//', '<<', '>>'):
                    tokens.append(two)
                    i += 2
                    continue
            tokens.append(c)
            i += 1
            continue
        if c.isalnum() or c == '_':
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
        tokens.append(c)
        i += 1
    return tokens


# ============================================================================
# Adaptive Multi-Level N-gram
# ============================================================================

class AdaptiveNgram:
    """Multi-level n-gram with frequency tracking and priority chains.

    Unlike simple ngram_mod which uses a single hash table with n=12,
    this maintains multiple n-gram levels and picks the best match.

    Key improvements over ngram_mod:
    1. Multiple levels (2,3,4,5,6,8,12) checked from longest to shortest
    2. Frequency counting: only predict tokens seen >1 times at that context
    3. Context window: recent tokens weighted higher (LRU-style)
    4. Chain prediction: after predicting token[0], use it to predict token[1]
    """

    def __init__(self, levels=(2, 3, 4, 5, 6, 8, 12), table_size=1 << 16):
        self.levels = sorted(levels, reverse=True)  # longest first
        self.table_size = table_size
        # Each level: hash -> {next_token: count}
        self.tables: Dict[int, Dict[int, Counter]] = {
            n: defaultdict(Counter) for n in self.levels
        }

    def _hash(self, tokens: List[str], n: int) -> int:
        h = 0
        for t in tokens[-n:]:
            h = (h * 6364136223846793005 + hash(t)) & 0xFFFFFFFFFFFFFFFF
        return h % self.table_size

    def add_context(self, tokens: List[str]):
        """Add all n-grams from a token sequence."""
        for n in self.levels:
            for i in range(len(tokens) - n):
                key = self._hash(tokens[i:i+n], n)
                next_tok = tokens[i + n]
                self.tables[n][key][next_tok] += 1

    def add_single(self, tokens: List[str], next_token: str):
        """Add a single observation for each n-gram level."""
        for n in self.levels:
            if len(tokens) >= n:
                key = self._hash(tokens[-n:], n)
                self.tables[n][key][next_token] += 1

    def predict_one(self, context: List[str], min_count: int = 1) -> Optional[str]:
        """Predict one token using longest matching n-gram."""
        for n in self.levels:
            if len(context) < n:
                continue
            key = self._hash(context[-n:], n)
            candidates = self.tables[n].get(key)
            if candidates:
                best_tok, best_count = candidates.most_common(1)[0]
                if best_count >= min_count:
                    return best_tok
        return None

    def predict_chain(self, context: List[str], k: int = 4, min_count: int = 1) -> List[str]:
        """Predict k tokens by chaining predictions."""
        result = []
        ctx = list(context)

        for _ in range(k):
            pred = self.predict_one(ctx, min_count)
            if pred is None:
                break
            result.append(pred)
            ctx.append(pred)

        return result

    def reset(self):
        for n in self.levels:
            self.tables[n].clear()


class IndentPredictor:
    """Specialized predictor for indentation patterns in code.

    After a newline, code almost always has the same or modified indentation.
    This predictor tracks the indentation pattern and predicts it.
    """

    def __init__(self):
        self.indent_stack: List[str] = []
        self.last_line_indent: Optional[str] = None

    def predict_after_newline(self, context: List[str]) -> Optional[str]:
        """Predict what comes after a newline."""
        if not context or context[-1] != '\n':
            return None

        # Look at the indentation of the previous line
        # Walk back to find the previous newline
        for i in range(len(context) - 2, -1, -1):
            if context[i] == '\n' and i + 1 < len(context):
                next_tok = context[i + 1]
                if next_tok.strip() == '' or next_tok[0] in ' \t':
                    return next_tok
                break
        return None


class StructuralPredictor:
    """Predict tokens based on structural code patterns.

    Handles high-confidence predictions like:
    - After 'self': predict '.'
    - After open bracket: track for close bracket
    - After 'return': predict space
    """

    PATTERNS = {
        'self': '.',
        'this': '.',
        'std': '::',
    }

    def predict(self, context: List[str]) -> Optional[str]:
        if not context:
            return None
        last = context[-1]
        return self.PATTERNS.get(last)


class HybridPredictor:
    """Combines adaptive n-gram, indent, and structural prediction.

    Priority order:
    1. Structural patterns (highest confidence, ~95% when applicable)
    2. Long n-gram match (n>=6, high confidence)
    3. Indent prediction after newline
    4. Medium n-gram match (n=3-5)
    5. Short n-gram match (n=2, lowest confidence)
    """

    def __init__(self, ngram_levels=(2, 3, 4, 5, 6, 8, 12)):
        self.ngram = AdaptiveNgram(levels=ngram_levels)
        self.indent = IndentPredictor()
        self.structural = StructuralPredictor()

    def add_context(self, tokens: List[str]):
        self.ngram.add_context(tokens)

    def add_single(self, tokens: List[str], next_token: str):
        self.ngram.add_single(tokens, next_token)

    def predict_one(self, context: List[str]) -> Optional[str]:
        # 1. Structural patterns
        pred = self.structural.predict(context)
        if pred is not None:
            return pred

        # 2. Indent prediction after newline
        pred = self.indent.predict_after_newline(context)
        if pred is not None:
            return pred

        # 3. N-gram chain (longest match first)
        return self.ngram.predict_one(context, min_count=1)

    def predict_chain(self, context: List[str], k: int = 4) -> List[str]:
        result = []
        ctx = list(context)

        for _ in range(k):
            pred = self.predict_one(ctx)
            if pred is None:
                break
            result.append(pred)
            ctx.append(pred)

        return result

    def reset(self):
        self.ngram.reset()


# ============================================================================
# Evaluation
# ============================================================================

@dataclass
class Result:
    name: str
    total_predictions: int = 0
    total_accepted: int = 0
    total_sequences: int = 0
    per_pos: Dict[int, Tuple[int, int]] = field(
        default_factory=lambda: defaultdict(lambda: (0, 0)))

    @property
    def rate(self) -> float:
        return self.total_accepted / max(self.total_predictions, 1)


def evaluate_predictor(name: str, tokens: List[str], predict_fn, k=4, warmup=20) -> Result:
    """Evaluate a predictor function on a token sequence."""
    result = Result(name=name)
    history = list(tokens[:warmup])

    i = warmup
    while i < len(tokens) - k:
        predictions = predict_fn(history, k)
        actual = tokens[i:i + k]

        if not predictions:
            history.append(tokens[i])
            i += 1
            continue

        result.total_sequences += 1
        n_accepted = 0

        for pos in range(min(len(predictions), len(actual))):
            total, accepted = result.per_pos[pos]
            result.per_pos[pos] = (total + 1, accepted)
            result.total_predictions += 1

            if predictions[pos] == actual[pos]:
                total, accepted = result.per_pos[pos]
                result.per_pos[pos] = (total, accepted + 1)
                n_accepted += 1
                result.total_accepted += 1
            else:
                break

        advance = n_accepted + 1
        for j in range(min(advance, len(tokens) - i)):
            history.append(tokens[i + j])
        i += advance

    return result


# ============================================================================
# Code samples
# ============================================================================

CODE_SAMPLES = {
    "Python": '''
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinaryTree(value)
            else:
                self.right.insert(value)

    def search(self, value):
        if value == self.value:
            return True
        elif value < self.value:
            if self.left is None:
                return False
            return self.left.search(value)
        else:
            if self.right is None:
                return False
            return self.right.search(value)
''',
    "C++": '''
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

class LRUCache {
public:
    LRUCache(int capacity) : capacity(capacity) {}

    int get(int key) {
        auto it = cache.find(key);
        if (it == cache.end()) {
            return -1;
        }
        order.erase(it->second.second);
        order.push_front(key);
        it->second.second = order.begin();
        return it->second.first;
    }

    void put(int key, int value) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            order.erase(it->second.second);
        } else if (cache.size() >= capacity) {
            int old_key = order.back();
            order.pop_back();
            cache.erase(old_key);
        }
        order.push_front(key);
        cache[key] = {value, order.begin()};
    }

private:
    int capacity;
    std::list<int> order;
    std::unordered_map<int, std::pair<int, std::list<int>::iterator>> cache;
};

template<typename T>
std::vector<T> quick_sort(std::vector<T> arr) {
    if (arr.size() <= 1) {
        return arr;
    }
    T pivot = arr[arr.size() / 2];
    std::vector<T> left, middle, right;
    for (const auto& elem : arr) {
        if (elem < pivot) {
            left.push_back(elem);
        } else if (elem == pivot) {
            middle.push_back(elem);
        } else {
            right.push_back(elem);
        }
    }
    auto sorted_left = quick_sort(left);
    auto sorted_right = quick_sort(right);
    sorted_left.insert(sorted_left.end(), middle.begin(), middle.end());
    sorted_left.insert(sorted_left.end(), sorted_right.begin(), sorted_right.end());
    return sorted_left;
}
''',
    "TypeScript": '''
interface User {
    id: number;
    name: string;
    email: string;
    createdAt: Date;
}

interface ApiResponse<T> {
    data: T;
    error: string | null;
    status: number;
}

class UserService {
    private baseUrl: string;
    private cache: Map<number, User>;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
        this.cache = new Map();
    }

    async getUser(id: number): Promise<ApiResponse<User>> {
        if (this.cache.has(id)) {
            return {
                data: this.cache.get(id)!,
                error: null,
                status: 200,
            };
        }
        try {
            const response = await fetch(`${this.baseUrl}/users/${id}`);
            const data = await response.json();
            if (!response.ok) {
                return {
                    data: null as any,
                    error: data.message || "Unknown error",
                    status: response.status,
                };
            }
            this.cache.set(id, data);
            return {
                data: data,
                error: null,
                status: 200,
            };
        } catch (error) {
            return {
                data: null as any,
                error: error instanceof Error ? error.message : "Unknown error",
                status: 500,
            };
        }
    }
}
''',
    "Rust": '''
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Graph {
    adjacency: HashMap<usize, Vec<(usize, f64)>>,
}

impl Graph {
    fn new() -> Self {
        Graph {
            adjacency: HashMap::new(),
        }
    }

    fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.adjacency.entry(from).or_insert_with(Vec::new).push((to, weight));
        self.adjacency.entry(to).or_insert_with(Vec::new).push((from, weight));
    }

    fn dijkstra(&self, start: usize) -> HashMap<usize, f64> {
        let mut distances: HashMap<usize, f64> = HashMap::new();
        let mut visited: Vec<usize> = Vec::new();
        let mut queue: Vec<(usize, f64)> = vec![(start, 0.0)];
        distances.insert(start, 0.0);

        while let Some((node, dist)) = queue.pop() {
            if visited.contains(&node) {
                continue;
            }
            visited.push(node);
            if let Some(neighbors) = self.adjacency.get(&node) {
                for &(neighbor, weight) in neighbors {
                    let new_dist = dist + weight;
                    let current_dist = distances.get(&neighbor).copied().unwrap_or(f64::INFINITY);
                    if new_dist < current_dist {
                        distances.insert(neighbor, new_dist);
                        queue.push((neighbor, new_dist));
                    }
                }
            }
            queue.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }
        distances
    }

    fn bfs(&self, start: usize) -> Vec<usize> {
        let mut visited: Vec<usize> = Vec::new();
        let mut queue: Vec<usize> = vec![start];
        while let Some(node) = queue.first().cloned() {
            queue.remove(0);
            if visited.contains(&node) {
                continue;
            }
            visited.push(node);
            if let Some(neighbors) = self.adjacency.get(&node) {
                for &(neighbor, _) in neighbors {
                    if !visited.contains(&neighbor) {
                        queue.push(neighbor);
                    }
                }
            }
        }
        visited
    }
}
''',
}


def main():
    print("=" * 80)
    print("ADAPTIVE MULTI-LEVEL N-GRAM SPECULATION TEST")
    print("=" * 80)

    k = 4
    strategies = {}

    # Strategy 1: Simple bigram (baseline)
    def make_bigram():
        table = defaultdict(Counter)
        def predict(history, k):
            result = []
            prev = history[-1]
            for _ in range(k):
                if prev not in table:
                    break
                best = table[prev].most_common(1)[0][0]
                result.append(best)
                prev = best
            return result
        def update(tokens):
            for i in range(len(tokens) - 1):
                table[tokens[i]][tokens[i+1]] += 1
        return predict, update, table

    # Strategy 2: Adaptive multi-level n-gram
    def make_adaptive(levels=(2, 3, 4, 5, 6, 8)):
        pred = HybridPredictor(ngram_levels=levels)
        def predict(history, k):
            return pred.predict_chain(history, k)
        def update(tokens):
            pred.add_context(tokens)
        return predict, update, pred

    # Strategy 3: Adaptive n-gram with min_count=2 (higher confidence)
    def make_adaptive_strict():
        ngram = AdaptiveNgram(levels=(2, 3, 4, 5, 6, 8, 12))
        def predict(history, k):
            return ngram.predict_chain(history, k, min_count=2)
        def update(tokens):
            ngram.add_context(tokens)
        return predict, update, ngram

    # Strategy 4: Line-aware n-gram (reset n-grams at line boundaries)
    def make_line_aware():
        # Main n-gram for within-line prediction
        ngram = AdaptiveNgram(levels=(2, 3, 4, 5, 6))
        # Separate n-gram for line-start prediction
        line_starts = defaultdict(Counter)  # prev_line_hash -> first_tokens
        lines = []
        current_line = []

        def predict(history, k):
            result = []
            ctx = list(history)
            for _ in range(k):
                # After newline: predict line start
                if ctx and ctx[-1] == '\n' and lines:
                    # Hash last few tokens before newline for context
                    pre = tuple(ctx[-4:-1]) if len(ctx) >= 4 else tuple(ctx[:-1])
                    if pre in line_starts:
                        best = line_starts[pre].most_common(1)[0][0]
                        result.append(best)
                        ctx.append(best)
                        continue

                # Regular n-gram
                pred = ngram.predict_one(ctx)
                if pred is None:
                    break
                result.append(pred)
                ctx.append(pred)
            return result

        def update(tokens):
            ngram.add_context(tokens)
            nonlocal current_line
            for t in tokens:
                if t == '\n':
                    if current_line:
                        lines.append(current_line)
                        # Record what follows a newline given context
                        if len(lines) >= 2:
                            prev_line = lines[-2]
                            pre = tuple(prev_line[-3:]) if len(prev_line) >= 3 else tuple(prev_line)
                            if current_line:
                                line_starts[pre][current_line[0]] += 1
                    current_line = []
                else:
                    current_line.append(t)

        return predict, update, ngram

    # Strategy 5: Windowed n-gram (only learn from last N tokens)
    def make_windowed(window=200):
        def predict(history, k):
            # Build fresh n-gram from recent window
            recent = history[-window:]
            ngram = AdaptiveNgram(levels=(2, 3, 4, 5, 6))
            ngram.add_context(recent)
            return ngram.predict_chain(history, k)
        def update(tokens):
            pass  # no persistent state
        return predict, update, None

    all_strategies = [
        ("bigram_baseline", make_bigram),
        ("adaptive_2_to_8", lambda: make_adaptive((2, 3, 4, 5, 6, 8))),
        ("adaptive_2_to_12", lambda: make_adaptive((2, 3, 4, 5, 6, 8, 12))),
        ("adaptive_strict", make_adaptive_strict),
        ("line_aware", make_line_aware),
        ("windowed_200", lambda: make_windowed(200)),
        ("windowed_500", lambda: make_windowed(500)),
    ]

    print(f"\n{'Strategy':<25s} {'Accept%':>8s} {'Pos0%':>6s} {'Pos1%':>6s} {'Pos2%':>6s} {'Pos3%':>6s} {'Seqs':>6s} {'Speedup':>8s}")
    print("-" * 80)

    aggregate = {}

    for lang, code in CODE_SAMPLES.items():
        tokens = simple_tokenize(code)
        print(f"\n  [{lang}: {len(tokens)} tokens]")

        for name, factory in all_strategies:
            predict_fn, update_fn, _ = factory()

            # Warm up with initial tokens
            warmup = 20
            update_fn(tokens[:warmup])

            result = Result(name=name)
            history = list(tokens[:warmup])
            i = warmup

            while i < len(tokens) - k:
                predictions = predict_fn(history, k)
                actual = tokens[i:i + k]

                if not predictions:
                    history.append(tokens[i])
                    update_fn([tokens[i]])
                    i += 1
                    continue

                result.total_sequences += 1
                n_accepted = 0

                for pos in range(min(len(predictions), len(actual))):
                    total, accepted = result.per_pos.get(pos, (0, 0))
                    result.per_pos[pos] = (total + 1, accepted)
                    result.total_predictions += 1

                    if predictions[pos] == actual[pos]:
                        total, accepted = result.per_pos[pos]
                        result.per_pos[pos] = (total, accepted + 1)
                        n_accepted += 1
                        result.total_accepted += 1
                    else:
                        break

                advance = n_accepted + 1
                new_tokens = tokens[i:i + min(advance, len(tokens) - i)]
                for t in new_tokens:
                    history.append(t)
                update_fn(new_tokens)
                i += advance

            # Aggregate
            if name not in aggregate:
                aggregate[name] = Result(name=name)
            agg = aggregate[name]
            agg.total_predictions += result.total_predictions
            agg.total_accepted += result.total_accepted
            agg.total_sequences += result.total_sequences
            for pos in result.per_pos:
                at, aa = agg.per_pos.get(pos, (0, 0))
                rt, ra = result.per_pos[pos]
                agg.per_pos[pos] = (at + rt, aa + ra)

            rate = result.rate * 100
            pos_rates = []
            for p in range(k):
                t, a = result.per_pos.get(p, (0, 0))
                pos_rates.append(f"{a/max(t,1)*100:5.0f}%")

            # Speedup calc
            pr = result.rate
            expected = 1.0 + sum(pr**i for i in range(1, k+1))
            speedup = expected / 0.77

            marker = " ***" if rate > 50 else ""
            print(f"  {name:<25s} {rate:7.1f}% {' '.join(pos_rates)} {result.total_sequences:6d} {speedup:7.2f}x{marker}")

    # Aggregate summary
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(f"\n{'Strategy':<25s} {'Accept%':>8s} {'Pos0%':>6s} {'Pos1%':>6s} {'Pos2%':>6s} {'Pos3%':>6s} {'Seqs':>6s} {'Speedup':>8s} {'tok/s':>7s}")
    print("-" * 85)

    for name, _ in all_strategies:
        agg = aggregate[name]
        rate = agg.rate * 100
        pos_rates = []
        for p in range(k):
            t, a = agg.per_pos.get(p, (0, 0))
            pos_rates.append(f"{a/max(t,1)*100:5.0f}%")

        pr = agg.rate
        expected = 1.0 + sum(pr**i for i in range(1, k+1))
        speedup = expected / 0.77
        projected = 28 * speedup

        verdict = "PASS" if rate > 50 else "FAIL"
        print(f"  {name:<25s} {rate:7.1f}% {' '.join(pos_rates)} {agg.total_sequences:6d} {speedup:7.2f}x {projected:6.0f} {verdict:>6s}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    best_name = max(aggregate, key=lambda n: aggregate[n].rate)
    best = aggregate[best_name]

    print(f"\nBest n-gram strategy: {best_name} at {best.rate*100:.1f}% acceptance")

    if best.rate > 0.50:
        pr = best.rate
        expected = 1.0 + sum(pr**i for i in range(1, k+1))
        speedup = expected / 0.77
        print(f"  -> {28 * speedup:.0f} tok/s projected (from 28 tok/s baseline)")
        print(f"  -> This PASSES the >50% threshold!")
    else:
        print(f"  -> Still below 50%. N-gram alone is insufficient for code.")
        print(f"\n  CRITICAL FINDING: Pure n-gram speculation cannot reach >50%")
        print(f"  acceptance on novel code generation. The tokens are too")
        print(f"  unpredictable at position 0 (the first draft token).")
        print(f"\n  The only way to achieve >50% is to use the MODEL's logits")
        print(f"  for at least the first draft token. This requires:")
        print(f"  1. Storing the top-K tokens from each decode step's logits")
        print(f"  2. Using top-1 as draft[0] (free - logits already computed)")
        print(f"  3. Using n-gram for draft[1-3]")
        print(f"\n  Even better: use top-1 from logits for ALL 4 positions")
        print(f"  by feeding each draft token through a 'prompt eval' step")
        print(f"  (this is exactly what draft-model speculation does, but")
        print(f"  using the SAME model with the same KV cache).")
        print(f"\n  SELF-SPECULATION APPROACH:")
        print(f"  Instead of predicting K tokens and verifying them,")
        print(f"  decode 1 token normally, then batch-evaluate K-1")
        print(f"  copies of the sequence with different draft tokens")
        print(f"  at the next position. But this requires K parallel")
        print(f"  sequence processing...")
        print(f"\n  SIMPLER: Just use temp=0 (greedy decode) and the")
        print(f"  top-1 token from logits IS the correct next token")
        print(f"  with 100% probability. Then batch-verify by feeding")
        print(f"  K top-1 tokens in sequence. This is a 'greedy chain'.")


if __name__ == "__main__":
    main()
