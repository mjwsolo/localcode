---
name: plan-task
description: Decompose a multi-step task into a written checklist BEFORE coding. Read-and-update the checklist between steps.
when_to_use: Task will likely take >4 tool calls — new feature, multi-file change, migration, debugging an unknown. Skip for trivial edits.
disable-model-invocation: true
---

Externalize the plan. Small quantized models (2-3 bit) drift on long
horizon tasks; a written plan becomes durable state they re-read.
Evidence: Agentless externalized-plan beat agent loops at comparable
compute (Xia et al. 2024); Voyager skill composition relies on the
same principle.

Format (output this as a markdown block BEFORE your first tool call):

```
Goal: <one sentence — what "done" looks like>

Steps:
1. [ ] <step 1 — small enough to verify independently>
       verify: <command or assertion that proves it worked>
2. [ ] <step 2>
       verify: <...>
3. [ ] <step 3>
       verify: <...>

Risks: <1-2 things that could go wrong>
```

Then execute ALL steps end-to-end in the SAME turn. After each step
completes, immediately move to the next step's tool call — do NOT
write a "progress so far / next steps" summary mid-task and stop. The
plan is a checklist for YOU to follow continuously, not a milestone
report for the user.

Only stop before finishing all steps if:
- A step's verify FAILS (then surface the failure and ask the user)
- The user explicitly asked you to stop after step N
- Tool execution returns REJECTED with a hard error you cannot recover from

Otherwise: keep going until every step is done. Save the final
summary for the very end, after the last step's verify passes.

Keep the plan ≤6 steps. If it's longer, the task is probably two
separate tasks — surface that to the user.

Anti-patterns:
- Re-planning after every step (wastes tokens).
- A 12-step plan that covers every imaginable branch.
- Writing "Current Progress / Next Steps" summary while steps remain.
- Treating completed-this-turn as completed-the-whole-task.
