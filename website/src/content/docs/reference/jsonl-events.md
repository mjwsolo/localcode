---
title: JSONL Events
description: The event stream emitted on stdout by localcode run --json.
---

`localcode run --goal "..." --json` outputs the agent's event stream as JSON Lines on stdout. Each line contains one JSON object. This lets editors and CI control localcode with code.

When `--json` is active, stdout stays clean. Rich output is turned off. The agent's raw ANSI output goes to `/dev/null`. The JSONL is written to a private copy of the original stdout file descriptor.

## Stream format

Every line is one event. The last line is always a `result` summary. Each line combines the event's `type` with its payload:

```json
{"type": "tool_start", "name": "read_file", "args": "src/timeutil.py", "index": "0"}
```

## Event types

| `type` | Payload | Meaning |
| --- | --- | --- |
| `thinking_start` | `reset` (`"true"`/`"false"`) | The model started thinking |
| `thinking_chunk` | `chunk` | Part of the hidden reasoning (limited to 2000 chars) |
| `thinking_peek` | `text` | Short preview of the current reasoning (120 chars) |
| `thinking_done` | `text` | The reasoning finished (limited to 8000 chars) |
| `stage` | `stage` | A named work stage that the UI can show |
| `stream_start` | - | The model started streaming its answer |
| `content` | `chunk`, `chars` | Part of the assistant output (limited to 2000 chars) |
| `tool_preview` | `name`, `chars`, `snippet` | A tool call is being built during streaming; its args are still growing |
| `tool_start` | `name`, `args`, `index` | A tool call was sent |
| `tool_result` | `name`, `args`, `index`, `result`, `error` | The call returned. `error` is `"true"`/`"false"`; `result` is limited to 4000 chars |
| `turn_tokens` | `prompt_tokens`, `completion_tokens`, `total_tokens` | Token use for one round. Values are **strings** |
| `notice` | `text` | A notice for the user, such as why a turn ended |
| `error` | `message` | An error that ends the turn (240 chars) |
| `done` | - | The turn finished |
| `result` | see below | Final event, always the last line |

Fields that look numeric (`index`, `chars`, and the three token counts) are output as **strings**. Convert them before doing arithmetic.

## The final `result` line

```json
{
  "type": "result",
  "status": "ok",
  "exit_code": 0,
  "reason": "completed",
  "final_text": "…",
  "tokens": {"prompt": 1200, "completion": 340, "total": 1540}
}
```

Here, the token counts *are* integers. The emitter adds them up from the `turn_tokens` events it received.

- `status` is `ok` when the turn's completion status is `completed`. Any other non-empty status (blocked, interrupted, stopped_early, error, incomplete) is reported as `incomplete`.
- `exit_code` is `0` for `ok`, `1` for `incomplete`, `124` for a timeout, and `130` for an interrupt.
- `reason` is the loop's blocked reason or exit reason, such as `completed`.

Read this line instead of joining the `content` chunks:

```sh
localcode run --goal "add a test for parse_config" --json | tail -1 | jq .
```

## Not included on stdout

The project audit log at `<project>/.localcode/events.jsonl` is a **separate** stream with a **different set of events**. It includes `turn_start`, `turn_end`, `round_start`, `round_end`, `auto_nudge`, server lifecycle records, and more. These events are written only to the file and never appear on stdout.

The project log is written whether or not you use `--json`. It is append-only. localcode never uploads it because no code path reads it for transmission. You can follow it during a run:

```sh
tail -f .localcode/events.jsonl | jq .
```
