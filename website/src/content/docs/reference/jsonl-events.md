---
title: JSONL Events
description: The event stream written to stdout by localcode run --json.
---

`localcode run --goal "..." --json` writes one JSON object per line to stdout, so an editor or CI job can drive localcode and read the result. Nothing else is written to stdout in this mode.

Each line has a `type` and the event's fields:

```json
{"type": "tool_start", "name": "read_file", "args": "src/timeutil.py", "index": "0"}
```

## Event types

| `type` | Fields | Meaning |
| --- | --- | --- |
| `thinking_start` | `reset` | The model started reasoning |
| `thinking_chunk` | `chunk` | Part of the hidden reasoning (up to 2000 chars) |
| `thinking_peek` | `text` | Short preview of the current reasoning (120 chars) |
| `thinking_done` | `text` | Reasoning finished (up to 8000 chars) |
| `stage` | `stage` | A named stage of the work |
| `stream_start` | | The model started its answer |
| `content` | `chunk`, `chars` | Part of the answer (up to 2000 chars) |
| `tool_preview` | `name`, `chars`, `snippet` | A tool call is still being written |
| `tool_start` | `name`, `args`, `index` | A tool call was sent |
| `tool_result` | `name`, `args`, `index`, `result`, `error` | The tool returned. `result` is cut at 4000 chars |
| `turn_tokens` | `prompt_tokens`, `completion_tokens`, `total_tokens` | Token use for one round |
| `notice` | `text` | A message for the user, such as why a turn ended |
| `error` | `message` | An error that ends the turn |
| `done` | | The turn finished |
| `result` | see below | Always the last line |

All field values are strings, including `index`, `chars`, `error` (`"true"` or `"false"`) and the token counts. Convert before doing arithmetic.

## The final `result` line

```json
{
  "type": "result",
  "status": "ok",
  "exit_code": 0,
  "reason": "completed",
  "final_text": "...",
  "tokens": {"prompt": 1200, "completion": 340, "total": 1540}
}
```

Here the token counts are integers.

- `status` is `ok` when the task completed, otherwise `incomplete`.
- `exit_code` is `0` for `ok`, `1` for `incomplete`, `124` for a timeout, `130` if interrupted.
- `reason` says why the run ended, for example `completed`.

Read this line rather than joining the `content` chunks:

```sh
localcode run --goal "add a test for parse_config" --json | tail -1 | jq .
```

## The project log is separate

`<project>/.localcode/events.jsonl` is a different, local-only stream with its own event types (`turn_start`, `turn_end`, `round_start`, server lifecycle and more). It is written whether or not you use `--json`, and it is never uploaded. Follow it during a run with:

```sh
tail -f .localcode/events.jsonl | jq .
```
