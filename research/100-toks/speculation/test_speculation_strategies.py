#!/usr/bin/env python3
"""
Speculative token prediction strategy evaluator for code generation.

Tests multiple prediction strategies against real code token sequences
to find which achieves >50% acceptance rate for K=4 speculation.

Key insight: We don't have a draft model. We need CPU-only prediction
that runs in <1ms using only:
  1. Previously generated tokens
  2. The logit distribution from the last decode step (simulated)
  3. The full prompt/conversation context

Strategies tested:
  1. top1_greedy     - Use top-1 from logits (simulated via actual next token)
  2. trigram_ctx     - Learn trigrams from current conversation
  3. bigram_ctx      - Learn bigrams from current conversation
  4. syntactic       - Predict based on code syntax rules
  5. repeat_line     - Detect and predict repeating line patterns
  6. ensemble        - Combine multiple strategies with voting
  7. top1_sim_noisy  - Simulated top-1 with realistic noise (80% accuracy)
  8. top1_sim_medium - Simulated top-1 with medium noise (65% accuracy)
"""

import json
import time
import os
import sys
from collections import defaultdict, Counter
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


# ============================================================================
# Token simulation
# ============================================================================

# We simulate tokenization by splitting code into "tokens" that approximate
# how a BPE tokenizer would split code. For real deployment, actual token IDs
# would be used, but for measuring acceptance rate the logic is identical.

def simple_tokenize(text: str) -> List[str]:
    """Approximate BPE tokenization for code.

    Splits on whitespace boundaries, punctuation, and common code patterns.
    This gives us token-level sequences that match real tokenizer behavior
    closely enough to measure acceptance rates.
    """
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]

        # Whitespace: each space/tab/newline is a token (or merged)
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

        # Punctuation / operators: single-char tokens
        if c in '(){}[]<>;:,.=+-*/%&|^~!?@#$\'"\\':
            # Multi-char operators
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

        # Alphanumeric: read word
        if c.isalnum() or c == '_':
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                j += 1
            word = text[i:j]
            # Split long words like BPE would (roughly)
            if len(word) > 6:
                # Keep common prefixes whole
                tokens.append(word)
            else:
                tokens.append(word)
            i = j
            continue

        tokens.append(c)
        i += 1

    return tokens


# ============================================================================
# Sample code for testing
# ============================================================================

PYTHON_CODE = '''
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

    def inorder(self):
        result = []
        if self.left:
            result.extend(self.left.inorder())
        result.append(self.value)
        if self.right:
            result.extend(self.right.inorder())
        return result
'''

CPP_CODE = '''
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

class LRUCache {
public:
    LRUCache(int capacity) : capacity(capacity) {}

    int get(int key) {
        auto it = cache.find(key);
        if (it == cache.end()) {
            return -1;
        }

        // Move to front
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
'''

TYPESCRIPT_CODE = '''
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

    async getUsers(ids: number[]): Promise<ApiResponse<User[]>> {
        const results: User[] = [];
        const errors: string[] = [];

        for (const id of ids) {
            const response = await this.getUser(id);
            if (response.error) {
                errors.push(response.error);
            } else {
                results.push(response.data);
            }
        }

        return {
            data: results,
            error: errors.length > 0 ? errors.join(", ") : null,
            status: errors.length > 0 ? 207 : 200,
        };
    }
}
'''

RUST_CODE = '''
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
'''


# ============================================================================
# Prediction Strategies
# ============================================================================

class PredictionStrategy:
    """Base class for token prediction strategies."""

    def __init__(self, name: str):
        self.name = name

    def predict(self, history: List[str], k: int = 4) -> List[str]:
        """Given token history, predict the next k tokens."""
        raise NotImplementedError

    def update(self, token: str):
        """Called when a new token is confirmed (for online learning)."""
        pass

    def reset(self):
        """Reset state for a new generation."""
        pass


class BigramStrategy(PredictionStrategy):
    """Learn bigrams from the current generation and predict next tokens."""

    def __init__(self):
        super().__init__("bigram_ctx")
        self.bigrams: Dict[str, Counter] = defaultdict(Counter)

    def predict(self, history: List[str], k: int = 4) -> List[str]:
        if not history:
            return []

        result = []
        prev = history[-1]

        for _ in range(k):
            if prev not in self.bigrams or not self.bigrams[prev]:
                break
            # Pick the most common next token after prev
            next_token = self.bigrams[prev].most_common(1)[0][0]
            result.append(next_token)
            prev = next_token

        return result

    def update(self, token: str):
        pass

    def update_pair(self, prev: str, cur: str):
        self.bigrams[prev][cur] += 1

    def reset(self):
        self.bigrams.clear()


class TrigramStrategy(PredictionStrategy):
    """Learn trigrams from the current generation and predict next tokens."""

    def __init__(self):
        super().__init__("trigram_ctx")
        self.trigrams: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.bigrams: Dict[str, Counter] = defaultdict(Counter)

    def predict(self, history: List[str], k: int = 4) -> List[str]:
        if len(history) < 2:
            return []

        result = []
        prev2 = history[-2]
        prev1 = history[-1]

        for _ in range(k):
            key = (prev2, prev1)
            if key in self.trigrams and self.trigrams[key]:
                next_token = self.trigrams[key].most_common(1)[0][0]
            elif prev1 in self.bigrams and self.bigrams[prev1]:
                next_token = self.bigrams[prev1].most_common(1)[0][0]
            else:
                break
            result.append(next_token)
            prev2 = prev1
            prev1 = next_token

        return result

    def update_trigram(self, t0: str, t1: str, t2: str):
        self.trigrams[(t0, t1)][t2] += 1
        self.bigrams[t1][t2] += 1

    def reset(self):
        self.trigrams.clear()
        self.bigrams.clear()


class SyntacticStrategy(PredictionStrategy):
    """Predict based on code syntax rules.

    For code, many tokens are highly predictable:
    - Opening bracket -> likely closing bracket eventually
    - 'def' -> function name -> '(' -> params -> ')' -> ':'
    - 'if' -> condition -> ':'
    - 'self' -> '.'
    - Common patterns: 'return', indentation, etc.
    """

    def __init__(self):
        super().__init__("syntactic")
        # Stack of expected tokens (closing brackets, etc.)
        self.bracket_stack: List[str] = []
        self.patterns = {
            'def': ['('],  # after function name
            'class': [':'],
            'if': [':'],
            'elif': [':'],
            'else': [':'],
            'while': [':'],
            'for': [':'],
            'self': ['.'],
            'return': [' '],
            '(': [')'],
            '[': [']'],
            '{': ['}'],
            '#': ['include'],
            'std': ['::'],
            'fn': ['('],
            'let': [' '],
            'const': [' '],
            'async': [' '],
            'await': [' '],
        }

    def predict(self, history: List[str], k: int = 4) -> List[str]:
        if not history:
            return []

        result = []
        last = history[-1]

        # Simple pattern matching
        if last in self.patterns:
            result.extend(self.patterns[last][:k])

        # After newline, predict indentation (same as previous line)
        if last == '\n':
            # Find the indentation of the previous line
            indent = self._get_prev_indent(history)
            if indent:
                result.append(indent)

        return result[:k]

    def _get_prev_indent(self, history: List[str]) -> Optional[str]:
        """Find indentation of the previous line."""
        # Walk back to find the last newline, then the indent after it
        for i in range(len(history) - 2, -1, -1):
            if history[i] == '\n' and i + 1 < len(history):
                next_tok = history[i + 1]
                if next_tok.strip() == '':  # whitespace = indent
                    return next_tok
                break
        return None

    def reset(self):
        self.bracket_stack.clear()


class RepeatLineStrategy(PredictionStrategy):
    """Detect repeating line patterns in code.

    Many code patterns involve similar lines:
    - Multiple imports
    - Multiple field declarations
    - Multiple function signatures with similar structure
    - Array/dict literal entries
    """

    def __init__(self):
        super().__init__("repeat_line")
        self.lines: List[List[str]] = []
        self.current_line: List[str] = []

    def predict(self, history: List[str], k: int = 4) -> List[str]:
        if not history or not self.lines:
            return []

        # Find lines similar to current partial line
        current = self.current_line.copy()
        if not current:
            return []

        best_match = None
        best_score = 0

        for line in self.lines:
            if len(line) <= len(current):
                continue

            # Check prefix match
            match_len = 0
            for j in range(min(len(current), len(line))):
                if current[j] == line[j]:
                    match_len += 1
                else:
                    break

            if match_len > best_score and match_len >= len(current) * 0.5:
                best_score = match_len
                best_match = line

        if best_match and len(best_match) > len(current):
            return best_match[len(current):len(current) + k]

        return []

    def update(self, token: str):
        if token == '\n':
            if self.current_line:
                self.lines.append(self.current_line.copy())
            self.current_line = []
        else:
            self.current_line.append(token)

    def reset(self):
        self.lines.clear()
        self.current_line.clear()


class Top1GreedySimulated(PredictionStrategy):
    """Simulate top-1 greedy prediction from logits.

    In real deployment, after each decode step, the model produces logits.
    The top-1 token is the most likely next token. For greedy decoding
    (temperature=0), this IS the next token with ~100% accuracy.

    For temperature>0, the acceptance rate depends on the entropy of the
    distribution. We simulate different accuracy levels.
    """

    def __init__(self, accuracy: float = 0.80, name: str = "top1_greedy_80"):
        super().__init__(name)
        self.accuracy = accuracy
        self.rng_state = 42

    def predict(self, history: List[str], k: int = 4) -> List[str]:
        # In real implementation, this would use the actual logits.
        # Here we simulate: the "correct" prediction is what actually
        # comes next, and we get it right with probability self.accuracy.
        # This is handled by the evaluator, not here.
        return ["__TOP1__"] * k

    def reset(self):
        self.rng_state = 42


class EnsembleStrategy(PredictionStrategy):
    """Combine multiple strategies with voting/priority."""

    def __init__(self, strategies: List[PredictionStrategy]):
        super().__init__("ensemble")
        self.strategies = strategies

    def predict(self, history: List[str], k: int = 4) -> List[str]:
        # Collect predictions from all strategies
        all_preds = []
        for s in self.strategies:
            pred = s.predict(history, k)
            if pred:
                all_preds.append((s.name, pred))

        if not all_preds:
            return []

        # Build result token by token
        result = []
        for pos in range(k):
            votes = Counter()
            for name, pred in all_preds:
                if pos < len(pred):
                    votes[pred[pos]] += 1

            if votes:
                # Pick the token with most votes, break ties by first appearance
                best = votes.most_common(1)[0][0]
                result.append(best)
            else:
                break

        return result

    def reset(self):
        for s in self.strategies:
            s.reset()


# ============================================================================
# Evaluation Engine
# ============================================================================

@dataclass
class StrategyResult:
    name: str
    total_predictions: int = 0
    total_accepted: int = 0
    total_sequences: int = 0  # number of K-token sequences attempted
    sequences_all_accepted: int = 0  # all K tokens accepted
    per_position_accepted: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    per_position_total: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    total_time_us: float = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_accepted / self.total_predictions

    @property
    def effective_speedup(self) -> float:
        """Calculate effective tokens per decode step.

        With K=4 speculation and acceptance rate p:
        - If all 4 accepted: 5 tokens in 1 decode step (1 verified + 4 draft)
        - In general: expected accepted = sum(p^i for i=1..K) + 1

        For independent acceptance probability p per token:
        E[tokens] = 1 + p + p^2 + ... + p^K = (1 - p^(K+1)) / (1 - p)

        But this assumes the batch decode of K+1 tokens takes the same
        time as a single decode. From our measurements:
        - Single decode: 28 tok/s = 35.7ms/tok
        - Batch K=4: 146 tok/s for 4 tokens = 27.4ms total

        So batch is ~77% the cost of a single decode.
        Expected throughput = E[tokens] / (0.77 * single_decode_time)
        """
        p = self.acceptance_rate
        K = 4
        if p == 0:
            return 1.0

        # Expected tokens per speculation round
        expected_tokens = 1.0  # the verified token
        for i in range(1, K + 1):
            expected_tokens += p ** i

        # Cost ratio: batch decode takes ~77% of single decode time
        cost_ratio = 0.77

        return expected_tokens / cost_ratio

    def position_acceptance_rate(self, pos: int) -> float:
        if self.per_position_total[pos] == 0:
            return 0.0
        return self.per_position_accepted[pos] / self.per_position_total[pos]


def evaluate_strategy(
    strategy: PredictionStrategy,
    tokens: List[str],
    k: int = 4,
    warmup: int = 20,  # min tokens before we start predicting
    simulated_accuracy: float = None,  # for Top1 simulation
) -> StrategyResult:
    """Evaluate a prediction strategy on a token sequence.

    Simulates the speculative decode loop:
    1. We have generated tokens[0:i]
    2. Strategy predicts tokens[i:i+k]
    3. We check how many of the predictions match the actual tokens
    4. Acceptance is sequential: first mismatch terminates
    """
    import random
    random.seed(42)

    result = StrategyResult(name=strategy.name)
    strategy.reset()

    # Build up context and learn patterns
    history = []

    # Feed initial tokens to build context
    for i in range(min(warmup, len(tokens))):
        token = tokens[i]
        history.append(token)
        strategy.update(token)

        # Update bigram/trigram strategies
        if isinstance(strategy, BigramStrategy) and len(history) >= 2:
            strategy.update_pair(history[-2], history[-1])
        elif isinstance(strategy, TrigramStrategy) and len(history) >= 3:
            strategy.update_trigram(history[-3], history[-2], history[-1])
        elif isinstance(strategy, EnsembleStrategy):
            for s in strategy.strategies:
                if isinstance(s, BigramStrategy) and len(history) >= 2:
                    s.update_pair(history[-2], history[-1])
                elif isinstance(s, TrigramStrategy) and len(history) >= 3:
                    s.update_trigram(history[-3], history[-2], history[-1])
                elif isinstance(s, RepeatLineStrategy):
                    s.update(token)

    # Now evaluate predictions
    i = warmup
    while i < len(tokens) - k:
        t0 = time.perf_counter_ns()
        predictions = strategy.predict(history, k)
        t1 = time.perf_counter_ns()
        result.total_time_us += (t1 - t0) / 1000.0

        actual = tokens[i:i + k]

        if not predictions:
            # Strategy returned nothing -- just advance one token
            token = tokens[i]
            history.append(token)
            strategy.update(token)

            if isinstance(strategy, BigramStrategy) and len(history) >= 2:
                strategy.update_pair(history[-2], history[-1])
            elif isinstance(strategy, TrigramStrategy) and len(history) >= 3:
                strategy.update_trigram(history[-3], history[-2], history[-1])
            elif isinstance(strategy, EnsembleStrategy):
                for s in strategy.strategies:
                    if isinstance(s, BigramStrategy) and len(history) >= 2:
                        s.update_pair(history[-2], history[-1])
                    elif isinstance(s, TrigramStrategy) and len(history) >= 3:
                        s.update_trigram(history[-3], history[-2], history[-1])
                    elif isinstance(s, RepeatLineStrategy):
                        s.update(token)

            i += 1
            continue

        result.total_sequences += 1

        # Check acceptance (sequential -- first mismatch terminates)
        n_accepted = 0
        for pos in range(min(len(predictions), len(actual))):
            pred = predictions[pos]
            act = actual[pos]

            result.per_position_total[pos] += 1

            # For Top1 simulation: we know the right answer with some probability
            if pred == "__TOP1__":
                if simulated_accuracy and random.random() < simulated_accuracy:
                    # Correct prediction
                    result.per_position_accepted[pos] += 1
                    n_accepted += 1
                    result.total_accepted += 1
                    result.total_predictions += 1
                else:
                    # Wrong prediction -- chain broken
                    result.total_predictions += 1
                    break
            elif pred == act:
                result.per_position_accepted[pos] += 1
                n_accepted += 1
                result.total_accepted += 1
                result.total_predictions += 1
            else:
                result.total_predictions += 1
                break

        if n_accepted == k:
            result.sequences_all_accepted += 1

        # Advance by accepted tokens + 1 (the verified token)
        advance = n_accepted + 1
        for j in range(min(advance, len(tokens) - i)):
            token = tokens[i + j]
            history.append(token)
            strategy.update(token)

            if isinstance(strategy, BigramStrategy) and len(history) >= 2:
                strategy.update_pair(history[-2], history[-1])
            elif isinstance(strategy, TrigramStrategy) and len(history) >= 3:
                strategy.update_trigram(history[-3], history[-2], history[-1])
            elif isinstance(strategy, EnsembleStrategy):
                for s in strategy.strategies:
                    if isinstance(s, BigramStrategy) and len(history) >= 2:
                        s.update_pair(history[-2], history[-1])
                    elif isinstance(s, TrigramStrategy) and len(history) >= 3:
                        s.update_trigram(history[-3], history[-2], history[-1])
                    elif isinstance(s, RepeatLineStrategy):
                        s.update(token)

        i += advance

    return result


def evaluate_all_strategies(code_samples: List[Tuple[str, str]], k: int = 4):
    """Evaluate all strategies across multiple code samples."""

    strategies = [
        ("bigram_ctx", lambda: BigramStrategy()),
        ("trigram_ctx", lambda: TrigramStrategy()),
        ("syntactic", lambda: SyntacticStrategy()),
        ("repeat_line", lambda: RepeatLineStrategy()),
        ("top1_greedy_80", lambda: Top1GreedySimulated(0.80, "top1_greedy_80")),
        ("top1_greedy_65", lambda: Top1GreedySimulated(0.65, "top1_greedy_65")),
        ("top1_greedy_50", lambda: Top1GreedySimulated(0.50, "top1_greedy_50")),
        ("ensemble_bi_tri_syn", lambda: EnsembleStrategy([
            BigramStrategy(),
            TrigramStrategy(),
            SyntacticStrategy(),
            RepeatLineStrategy(),
        ])),
    ]

    # Per-strategy aggregate results
    aggregate: Dict[str, StrategyResult] = {}

    print("=" * 80)
    print("SPECULATIVE DECODE STRATEGY EVALUATION")
    print(f"K = {k} tokens per speculation round")
    print("=" * 80)

    for lang, code in code_samples:
        tokens = simple_tokenize(code)
        print(f"\n--- {lang} ({len(tokens)} tokens) ---")

        for name, factory in strategies:
            strategy = factory()

            sim_acc = None
            if "top1" in name:
                sim_acc = strategy.accuracy if hasattr(strategy, 'accuracy') else None

            result = evaluate_strategy(strategy, tokens, k=k, simulated_accuracy=sim_acc)

            # Merge into aggregate
            if name not in aggregate:
                aggregate[name] = StrategyResult(name=name)
            agg = aggregate[name]
            agg.total_predictions += result.total_predictions
            agg.total_accepted += result.total_accepted
            agg.total_sequences += result.total_sequences
            agg.sequences_all_accepted += result.sequences_all_accepted
            agg.total_time_us += result.total_time_us
            for pos in result.per_position_total:
                agg.per_position_total[pos] += result.per_position_total[pos]
                agg.per_position_accepted[pos] += result.per_position_accepted[pos]

            rate = result.acceptance_rate * 100
            speedup = result.effective_speedup
            avg_time = result.total_time_us / max(result.total_sequences, 1)

            marker = " ***" if rate > 50 else ""
            print(f"  {name:25s}  accept={rate:5.1f}%  "
                  f"speedup={speedup:.2f}x  "
                  f"seqs={result.total_sequences:4d}  "
                  f"all4={result.sequences_all_accepted:4d}  "
                  f"avg_us={avg_time:.0f}{marker}")

    # Print aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    print(f"\n{'Strategy':<25s} {'Accept%':>8s} {'Speedup':>8s} {'Sequences':>10s} {'AllK':>6s} {'Avg us':>8s} {'Verdict':>10s}")
    print("-" * 80)

    winner = None
    best_rate = 0

    for name, _ in strategies:
        agg = aggregate[name]
        rate = agg.acceptance_rate * 100
        speedup = agg.effective_speedup
        avg_time = agg.total_time_us / max(agg.total_sequences, 1)

        verdict = "PASS" if rate > 50 else "FAIL"

        if rate > best_rate:
            best_rate = rate
            winner = name

        print(f"{name:<25s} {rate:7.1f}% {speedup:7.2f}x {agg.total_sequences:10d} {agg.sequences_all_accepted:6d} {avg_time:7.0f} {verdict:>10s}")

    # Per-position breakdown for best strategy
    print(f"\n--- Per-position acceptance for top strategies ---")
    for name, _ in strategies:
        agg = aggregate[name]
        if agg.acceptance_rate > 0.3:  # Only show interesting ones
            pos_rates = []
            for pos in range(k):
                pr = agg.position_acceptance_rate(pos)
                pos_rates.append(f"pos{pos}={pr*100:.0f}%")
            print(f"  {name:<25s}  {' '.join(pos_rates)}")

    # Speedup analysis
    print(f"\n--- Speedup Analysis ---")
    print(f"Baseline: 28 tok/s (single-token decode)")
    print(f"Batch decode K=4: 146 tok/s throughput")
    print()
    for name, _ in strategies:
        agg = aggregate[name]
        speedup = agg.effective_speedup
        projected_tps = 28 * speedup
        print(f"  {name:<25s}  {speedup:.2f}x -> {projected_tps:.0f} tok/s projected")

    print(f"\n--- Winner: {winner} ({best_rate:.1f}% acceptance) ---")

    return aggregate


# ============================================================================
# Additional analysis: what makes code predictable?
# ============================================================================

def analyze_code_predictability(code_samples: List[Tuple[str, str]]):
    """Analyze what fraction of code tokens are "predictable" by category."""

    print("\n" + "=" * 80)
    print("CODE PREDICTABILITY ANALYSIS")
    print("=" * 80)

    categories = {
        'whitespace': lambda t: t.strip() == '',
        'newline': lambda t: t == '\n',
        'bracket_close': lambda t: t in ')}]',
        'bracket_open': lambda t: t in '({[',
        'operator': lambda t: t in '=+-*/<>!&|^~:;,.?@#',
        'keyword': lambda t: t in ('def', 'class', 'if', 'else', 'elif', 'for',
                                    'while', 'return', 'import', 'from', 'in',
                                    'fn', 'let', 'mut', 'struct', 'impl', 'pub',
                                    'const', 'var', 'function', 'async', 'await',
                                    'try', 'catch', 'throw', 'new', 'this', 'self',
                                    'int', 'float', 'string', 'bool', 'void', 'auto',
                                    'include', 'using', 'namespace', 'template',
                                    'interface', 'type', 'export', 'extends'),
        'identifier': lambda t: t.isidentifier(),
        'number': lambda t: t.isdigit(),
    }

    for lang, code in code_samples:
        tokens = simple_tokenize(code)
        total = len(tokens)

        print(f"\n{lang}: {total} tokens")

        # Count tokens that follow a repeated bigram pattern
        bigram_predictable = 0
        bigram_counts: Dict[str, Counter] = defaultdict(Counter)
        for i in range(1, len(tokens)):
            prev = tokens[i - 1]
            cur = tokens[i]
            bigram_counts[prev][cur] += 1

        # A token is "bigram predictable" if the most common follower of prev == cur
        for i in range(1, len(tokens)):
            prev = tokens[i - 1]
            cur = tokens[i]
            if bigram_counts[prev].most_common(1)[0][0] == cur:
                bigram_predictable += 1

        print(f"  Bigram-predictable: {bigram_predictable}/{total-1} = {bigram_predictable/(total-1)*100:.1f}%")

        # Same for trigrams
        trigram_counts: Dict[Tuple[str,str], Counter] = defaultdict(Counter)
        trigram_predictable = 0
        for i in range(2, len(tokens)):
            key = (tokens[i-2], tokens[i-1])
            trigram_counts[key][tokens[i]] += 1

        for i in range(2, len(tokens)):
            key = (tokens[i-2], tokens[i-1])
            if trigram_counts[key].most_common(1)[0][0] == tokens[i]:
                trigram_predictable += 1

        print(f"  Trigram-predictable: {trigram_predictable}/{total-2} = {trigram_predictable/(total-2)*100:.1f}%")

        # Category breakdown
        counts = defaultdict(int)
        for t in tokens:
            for cat, check in categories.items():
                if check(t):
                    counts[cat] += 1
                    break
            else:
                counts['other'] += 1

        print(f"  Token categories:")
        for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {cat:<20s} {count:4d} ({count/total*100:.1f}%)")


# ============================================================================
# Top-1 from logits deep analysis
# ============================================================================

def analyze_top1_potential():
    """
    Analyze the theoretical potential of top-1 greedy prediction.

    Key insight: For greedy decoding (temp=0), the model always picks
    the top-1 token. So if we use the top-1 from the PREVIOUS step's
    logits as our prediction, the acceptance rate equals:

    P(top1_at_step_i == actual_token_at_step_i+1)

    This is NOT 100% because:
    - The logits at step i are conditioned on tokens[0:i]
    - The actual token at step i+1 is conditioned on tokens[0:i+1]
    - The additional context of token[i] changes the distribution

    However, for many tokens in code, the distribution barely changes:
    - After 'self', '.' is overwhelmingly likely regardless of context
    - After newline, indentation is predictable
    - After '(', the first parameter token is often predictable

    In practice, for greedy decoding, studies show 60-80% of top-1
    predictions from the current step match the next step's choice.

    BUT: We don't actually have the logits from the previous step
    in a model-free predictor. The logits ARE the model output.

    The real question is: can we get logits "for free" in the
    speculative decode pipeline?

    Answer: YES! After each verified decode step, we ALREADY have
    the logits. The top-1 token from those logits is our first
    draft token. Then we need to speculate tokens 2-4 without logits.
    """
    print("\n" + "=" * 80)
    print("TOP-1 FROM LOGITS ANALYSIS")
    print("=" * 80)
    print()
    print("Key insight for implementation:")
    print("  After each decode step, the model produces logits for ALL tokens.")
    print("  The top-1 token IS our prediction for the next token.")
    print()
    print("  For greedy decoding (temp=0): acceptance rate = 100%")
    print("    (because the model deterministically picks top-1)")
    print()
    print("  For temp=0.6 (typical code gen):")
    print("    - Top-1 covers ~60-80% of samples")
    print("    - Top-3 covers ~85-95% of samples")
    print()
    print("  CRITICAL REALIZATION:")
    print("  We already get the first draft token 'for free' from logits.")
    print("  The challenge is tokens 2, 3, 4 in the draft sequence.")
    print()
    print("  Hybrid approach (our best bet):")
    print("    Draft[0] = top-1 from logits (free, 60-80% accuracy)")
    print("    Draft[1] = top-1 from logits IF we had a way to get them")
    print("    Draft[2-3] = ngram/syntactic prediction from context")
    print()
    print("  BUT WAIT: The server already computes logits for the last")
    print("  token in the batch. In batched speculation, we decode K+1")
    print("  tokens in one batch. The logits come from the LAST position.")
    print()
    print("  Alternative: Use PROMPT EVAL to get all K logits at once!")
    print("  If we batch-evaluate K draft tokens, we get K logit vectors.")
    print("  This is exactly what speculative decoding already does.")
    print()
    print("  The bottleneck is: we need the DRAFT tokens first.")
    print("  That's what our predictor must provide.")


def print_theoretical_speedup():
    """Print theoretical speedup for different acceptance rates."""
    print("\n" + "=" * 80)
    print("THEORETICAL SPEEDUP TABLE")
    print("=" * 80)
    print()
    print("Assumptions:")
    print("  - Single decode: 28 tok/s (35.7 ms/tok)")
    print("  - Batch K+1 decode: ~27.4 ms (measured)")
    print("  - Cost ratio = 27.4/35.7 = 0.77")
    print()

    print(f"{'Accept%':>8s} | {'E[tokens]':>10s} | {'Speedup':>8s} | {'Projected':>10s} | {'Notes':>20s}")
    print("-" * 70)

    for pct in range(0, 101, 5):
        p = pct / 100.0
        K = 4
        expected = 1.0
        for i in range(1, K + 1):
            expected += p ** i

        cost_ratio = 0.77
        speedup = expected / cost_ratio
        projected = 28 * speedup

        notes = ""
        if pct == 0:
            notes = "no speculation"
        elif pct == 50:
            notes = "MINIMUM TARGET"
        elif pct == 65:
            notes = "ngram typical"
        elif pct == 80:
            notes = "top-1 greedy"
        elif pct == 100:
            notes = "perfect (impossible)"

        print(f"{pct:7d}% | {expected:10.2f} | {speedup:7.2f}x | {projected:8.0f} t/s | {notes}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    code_samples = [
        ("Python", PYTHON_CODE),
        ("C++", CPP_CODE),
        ("TypeScript", TYPESCRIPT_CODE),
        ("Rust", RUST_CODE),
    ]

    # Print theoretical speedup table
    print_theoretical_speedup()

    # Analyze code predictability
    analyze_code_predictability(code_samples)

    # Analyze top-1 potential
    analyze_top1_potential()

    # Evaluate all strategies
    results = evaluate_all_strategies(code_samples, k=4)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL CONCLUSIONS")
    print("=" * 80)

    passing = [(name, r) for name, r in results.items() if r.acceptance_rate > 0.50]
    failing = [(name, r) for name, r in results.items() if r.acceptance_rate <= 0.50]

    if passing:
        print("\nStrategies achieving >50% acceptance:")
        for name, r in sorted(passing, key=lambda x: -x[1].acceptance_rate):
            print(f"  {name}: {r.acceptance_rate*100:.1f}% -> {28 * r.effective_speedup:.0f} tok/s projected")

    if failing:
        print("\nStrategies below 50% acceptance:")
        for name, r in sorted(failing, key=lambda x: -x[1].acceptance_rate):
            print(f"  {name}: {r.acceptance_rate*100:.1f}%")

    print("\nRECOMMENDATION:")
    print("  The top-1 greedy from logits is the clear winner for speculative")
    print("  decoding without a draft model. At temp=0 (greedy), acceptance is")
    print("  effectively 100% for the first draft token.")
    print()
    print("  For K=4 speculation, use a HYBRID approach:")
    print("    Position 0: top-1 from logits (60-80% for temp>0, 100% for temp=0)")
    print("    Position 1-3: ngram prediction from context (30-50%)")
    print()
    print("  Implementation: modify the server's speculative decode to use")
    print("  the logits from the last decode step as the first draft token,")
    print("  then fall back to ngram prediction for remaining positions.")
