---
title: JSONL Events
description: The event stream emitted on stdout by localcode run --json.
---

`localcode run --goal "..." --json` emits the agent's event stream as JSON
Lines on stdout — one JSON object per line — so editors and CI can drive
localcode programmatically.

While `--json` is active, stdout is kept clean: Rich output is silenced, the
agent's raw ANSI writes are redirected to `/dev/null`, and the JSONL is written
to a private duplicate of the original stdout file descriptor.

## Where the stream comes from

Every line except the last is one call to the `OutputManager` event callback —
the same callback the TUI subscribes to. `run --json` installs the JSONL
emitter as that callback, so **the stdout vocabulary is exactly the set of
events `OutputManager` emits**, plus a terminal `result` line the emitter
writes itself.

Each line is the event's `type` merged with its payload:

```json
{"type": "tool_start", "name": "read_file", "args": "src/timeutil.py", "index": "0"}
```

## Event types

| `type` | Payload | Meaning |
| --- | --- | --- |
| `thinking_start` | `reset` (`"true"`/`"false"`) | The model began a thinking phase |
| `thinking_chunk` | `chunk` | A chunk of hidden reasoning (capped at 2000 chars) |
| `thinking_peek` | `text` | Short preview of the current reasoning (120 chars) |
| `thinking_done` | `text` | Reasoning finished (capped at 8000 chars) |
| `stage` | `stage` | A named work stage the UI can display |
| `stream_start` | — | The model started streaming its answer |
| `content` | `chunk`, `chars` | A chunk of assistant output (capped at 2000 chars) |
| `tool_preview` | `name`, `chars`, `snippet` | A tool call is forming mid-stream; args still growing |
| `tool_start` | `name`, `args`, `index` | A tool call was dispatched |
| `tool_result` | `name`, `args`, `index`, `result`, `error` | That call returned. `error` is `"true"`/`"false"`; `result` is capped at 4000 chars |
| `turn_tokens` | `prompt_tokens`, `completion_tokens`, `total_tokens` | Token usage for one round. Values are **strings** |
| `notice` | `text` | A user-facing notice, e.g. why a turn ended |
| `error` | `message` | A turn-ending error (240 chars) |
| `done` | — | The turn finished |
| `result` | see below | Terminal event, always the last line |

Numeric-looking fields (`index`, `chars`, the three token counts) are emitted
as **strings**; coerce them before arithmetic.

## The terminal `result` line

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

Here the token counts *are* integers — the emitter accumulates them from the
`turn_tokens` events it saw.

- `status` is `ok` when the turn's completion status was `completed`; any other
  non-empty status (blocked, interrupted, stopped_early, error, incomplete)
  reports `incomplete`.
- `exit_code` is `0` for `ok`, `1` for `incomplete`, `124` on timeout, `130` on
  interrupt.
- `reason` is the loop's blocked reason or exit reason, e.g. `completed`.

Read this line rather than reassembling `content` chunks:

```sh
localcode run --goal "add a test for parse_config" --json | tail -1 | jq .
```

## Not emitted on stdout

The project audit log — `<project>/.localcode/events.jsonl` — is a **separate**
stream with a **different vocabulary**. It carries `turn_start`, `turn_end`,
`round_start`, `round_end`, `auto_nudge`, server lifecycle records and more.
Those events go to the file, not through the `OutputManager` callback, so they
never appear on stdout.

This matters in practice: `run --json` cannot read the turn's final status off
its own stdout stream, which is why it reads the status the loop persisted on
the app object instead.

The project log is written whether or not you use `--json`, is append-only, and
never leaves the machine. Tail it alongside a run:

```sh
tail -f .localcode/events.jsonl | jq .
```

:::note[Fixtures are not the producer]
`tests/protocol_fixtures/*.jsonl` exercise a **consumer/parser** and contain a
hypothetical `run_start` / `round_start` / `turn_end` shape. They do not
describe what `run --json` writes. The table above is taken from the emitting
code.
:::
