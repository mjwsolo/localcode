# Remaining Components to Implement

## Ship-blocking (MUST HAVE):
- [x] 12 tools (8 model-facing + 4 harness-only)
- [x] Context assembler + token budget
- [x] File summarizer (AST-based)
- [ ] Output parser (robust code/search-replace/plan extraction)
- [ ] Safety layer (command blocking, file jailing)
- [x] Inference layer (Ollama with /api/generate)
- [ ] Streaming output in agent loop
- [x] System prompts (tiny, task-specific)
- [x] Intent classifier (rule-based + LLM fallback)
- [x] 13 feature executors (state machines)
- [x] Undo stack
- [x] Syntax checker (auto-run after writes)

## Should have (quality of life):
- [ ] Project indexer (symbol lookup, file stats)
- [ ] .gitignore filtering
- [x] Conversation history compression
- [x] Relevance finder (keyword + recency)
- [ ] Config file (user-facing)
- [ ] Session logging
- [x] Progress tracking for multi-step
- [ ] Diff preview before applying edits

## Nice to have (later):
- [x] Plan layer (Layer 3) for complex tasks
- [x] Orchestrator with replanning
- [ ] Model routing (small/large) — limited by 16GB RAM
- [ ] File watcher (auto-refresh index)
- [ ] LSP integration
- [ ] Embeddings-based search
- [ ] TUI with panels

## Token budgets per task type:
```
classify:  10 tokens
explain:   512
review:    1024
edit:      2048
create:    4096
plan:      1024
```
