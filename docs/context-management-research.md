# Context management for long-running local tasks — research + implementation

Source review of four production coding-agent harnesses, read directly from
their repositories, to fix the two failures we saw on long local builds:

1. **Churn** — the model re-reads files it already wrote, re-discovers the same
   "inconsistencies," and loops (`"Let me fix them systematically"` ×3).
2. **Misattribution** — after compaction the model thinks *"the user created
   these files"* / *"the user hasn't given me a task"* about its own work.

Harnesses read: **pi** (earendil-works/pi), **codex** (openai/codex),
**claude-code** (real production source), **opencode** (sst/opencode).

## What all four converge on

1. **Provenance = message role/type, never prose.** A past tool action is stored
   as a structured `assistant` tool-call + a separate `tool`/`tool_result`
   item keyed by id, round-tripped verbatim every turn. None of them ever
   flatten a past assistant action into `role:"user"` text in the live log.
   - codex: `FunctionCall` + role-less `FunctionCallOutput`, attribution derived
     from item *type* (`is_model_generated_item`, history.rs:722).
   - pi: `assistant`(toolCall blocks) + `toolResult` messages, pass through
     `convertToLlm` untouched (messages.ts:155).
   - claude-code: `assistant` `tool_use` + `user` `tool_result` keyed by
     `tool_use_id`; trimming swaps the *content string* but keeps the block+id.
   - opencode: structured parts keyed by `callID` (message-v2.ts:290).

2. **Compaction threshold is derived from the model's real context window — never
   a hardcoded constant.**
   - pi: `contextTokens > contextWindow - reserveTokens` (reserve 16384).
   - opencode: `usable = model.limit.input - min(20_000, maxOutput)` — window
     from the models.dev catalog per model (overflow.ts:10-34).
   - codex: fires on `token_limit_reached`.
   - claude-code: token estimate crosses a fraction of the model window.

3. **Keep the recent history by token budget, scaled to the window** — not a
   fixed message count. opencode keeps 25% of usable, clamped [2k, 8k]; pi keeps
   20k; claude-code slices at the boundary and keeps everything after verbatim.

4. **The summary is framed so it never reads as user-authored.**
   - codex: prefix *"Another language model produced this summary… you have
     access to the state of the tools it used."*
   - opencode: inject the summary as an **assistant** turn answering a synthetic
     user *"What did we do so far?"*; template is third-person/imperative.
   - pi: neutral third-person checkpoint (`## Progress → Done/[x]`).
   - claude-code: summary prompt splits "user's explicit requests" (§1) from
     "Files… **created**" attributed to "you" (§3), and §6 lists "All user
     messages **that are not tool results**" — the exact guard against treating
     a `tool_result` (wire-role user) as a user statement.

5. **Age/prune old tool-result payloads before a full summary** (cheap, no LLM):
   claude-code blanks old read-tool results to `"[Old tool result content
   cleared]"` keeping the block+id; opencode marks old tool output `compacted`
   past a protect threshold.

6. **A structured file-ops ledger, carried separately across compactions**: pi
   `<modified-files>`; opencode SQL todo table + pruning; codex WorldState diff.

## How localcode maps to this

| Pattern | localcode |
|---|---|
| Live tool results as `role:"tool"`+`tool_call_id` | ✅ `agent/loop.py` (never flattened) |
| Dynamic compaction threshold off `_target_num_ctx()` | ✅ `compaction.py:should_compact` (0.3.16 removed the stale hardcoded 12K/27K second pass) |
| Keep-recent scaled to window | ✅ `compaction.py:_keep_recent_for_window` (16GB≈6k → 128GB≈48k) |
| Summary framed as not-user-authored | ✅ `role:"system"` memo: *"the work already done — build on it, do NOT re-read"* |
| Age old tool results | ✅ `agent/context.py:_compact_old_tool_results` |
| "Files YOU created" ledger, anti-misattribution | ✅ `agent/context.py:build_progress_ledger` (header: *"NOT the user's work"*) |

## The bug we fixed in 0.3.16

localcode had **two** compaction systems. The good one (`compaction.py`, window-
aware, `role:"system"` framing) runs at the top of each round. A **stale second
pass** (`agent/context.py:_compact_messages`, fixed 12K/27K) ran at the bottom,
fired at <5% of a 256K window, crushed all but 8 messages into a 5-line
`role:"user"` note that said *"Completed: (none yet)"* — which the model read as
a user message with no task. Removed; the dynamic path governs alone.

## Candidate follow-ups (not yet done — evaluate per impact)

- Adopt opencode's assistant-turn summary framing (synthetic *"What did we do so
  far?"* → assistant answer) for even crisper provenance than the current
  `role:"system"` memo.
- Todo-reminder decay (claude-code): only re-surface the todo list after N turns
  since the last `todo_write`, instead of injecting the ledger every round.
- Sub-agent delegation: fresh-context teammate per sub-task returning a short
  result — the biggest *context-isolation* win for long local builds (memory-
  bound, so sequential by default; parallel only on big Macs via `-np`).
- Dynamic "N tokens left" reminder, scaled to the live window utilization.
