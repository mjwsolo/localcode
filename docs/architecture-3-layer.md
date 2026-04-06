# LocalCode 3-Layer Architecture

**Flat state machine covers 60% of tasks. 3-layer covers ~85%. The remaining 15% needs user guidance.**

## Architecture

```
USER INPUT → COMPLEXITY CLASSIFIER → simple? → Layer 1 (direct execution)
                                   → complex? → Layer 3 (plan) → Layer 2 (orchestrate) → Layer 1 (execute)
```

### Layer 1: EXECUTORS (13 features)
- Deterministic state machines
- Model generates content ONLY
- Harness owns tools + validation
- Each feature: GATHER → LLM_CALL → APPLY → VERIFY → FIX_LOOP

### Layer 2: ORCHESTRATOR
- Walks step list topologically
- Manages context window (CRITICAL for 16GB / small context)
- Handles replan on failure
- Parallelizes independent steps
- Passes output from step N as input to step N+1

### Layer 3: PLANNER
- Only activated for multi-step tasks
- 4 tool calls: glob + read + bash (parallel) + llm_call
- Outputs ordered step list with dependencies
- Show plan to user for big tasks before executing

## When to activate Layer 3

```python
def needs_planning(user_input, file_tree):
    # Multiple verbs: "add auth AND write tests AND update docs"
    if count_action_verbs(user_input) >= 2:
        return True
    # Broad scope keywords
    broad = ["authentication", "database", "API", "refactor all",
             "migrate", "set up", "implement"]
    if any(k in user_input.lower() for k in broad):
        return True
    # Multiple files mentioned
    if mentions_multiple_files(user_input):
        return True
    return False
```

## Context Windowing (make-or-break for small models)

```python
def build_context(step, dep_results, max_tokens=2048):
    budget = max_tokens
    parts = []

    # Priority 1: file being edited (truncate to relevant section if >100 lines)
    # Priority 2: outputs from dependency steps (summaries, not full files)
    # Priority 3: related file signatures (imports, function names)
```

Good context selection makes a 26B perform like a 70B.

## Replanning on Failure

When step N fails after retries:
1. llm_call with: original plan + error + completed steps
2. Model outputs revised remaining steps
3. Max 2 replans, then surface error to user

## Plan Templates (for common tasks)

Instead of asking model to plan from scratch, use hardcoded skeletons:

- **Add auth**: install deps → create user model → create auth routes → edit app → edit routes → tests
- **Add CRUD endpoint**: create model → create route → register route → tests
- **Add tests**: read source → generate tests → run tests → fix tests

Model fills in the DETAILS, harness provides the STRUCTURE.

## Engineering Priority Order

1. **Context windowing** — #1 make-or-break. Good context = 26B performs like 70B.
2. **Harness validation** — catch bad output before disk. Syntax, search-string, imports.
3. **Plan templates** — hardcoded skeletons for common tasks.
4. **Graceful degradation** — surface garbage cleanly, don't cascade failures.

## Confidence by Task Type

| Capability | Confidence |
|-----------|-----------|
| Single-file create/edit | High |
| Code review / explain | High |
| 3-5 step plans | Medium |
| 8+ step plans | Low — harness must compensate |
| Cross-file refactors | Medium — context windowing is bottleneck |
| Replanning after failure | Low — requires reasoning about what went wrong |
