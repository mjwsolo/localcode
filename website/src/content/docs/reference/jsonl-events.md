---
title: JSONL Events
description: The event stream emitted by localcode run --json.
---

`localcode run --goal "..." --json` emits the agent's event stream as JSON
Lines on stdout — one JSON object per line — so editors and CI can drive
localcode programmatically.

While `--json` is active, stdout is kept clean: Rich output is silenced and the
JSONL is written to a private duplicate of the original stdout file
descriptor.

## Shape

Every line has a `type`. The first line is `run_start` and carries
`schema_version`; the last line is always `result`.

```jsonl
{"type":"run_start","schema_version":"1.0","run_id":"r-001","localcode_version":"0.3.36"}
{"type":"turn_start"}
{"type":"round_start","round_idx":0}
{"type":"tool_call","name":"read_file","round_idx":0}
{"type":"tool_result","name":"read_file","status":"ok","round_idx":0}
{"type":"round_end","round_idx":0}
{"type":"turn_tokens","prompt_tokens":"1200","completion_tokens":"340","total_tokens":"1540"}
{"type":"turn_end","completion_status":"completed","loop_exit_reason":"completed"}
{"type":"result","status":"ok","exit_code":0,"reason":"completed","final_text":"Done.","tokens":{"prompt":1200,"completion":340,"total":1540}}
```

That example is the checked-in fixture
`tests/protocol_fixtures/success.jsonl`, which the protocol tests read.

## Event types

| `type` | Meaning |
| --- | --- |
| `run_start` | Stream opened. Carries `schema_version`, `run_id`, `localcode_version` |
| `turn_start` / `turn_end` | Turn boundaries. `turn_end` carries `completion_status` and `loop_exit_reason` |
| `round_start` / `round_end` | One model/tool round, indexed by `round_idx` |
| `tool_call` | The model called a tool: `name`, `round_idx` |
| `tool_result` | Result of that call: `name`, `status` (`ok` / `error`) |
| `turn_tokens` | Per-round token counts, accumulated for the final summary |
| `result` | Terminal event: `status`, `exit_code`, `reason`, `final_text`, `tokens` |

## Consuming it

Read the trailing `result` line rather than reassembling content deltas:

```sh
localcode run --goal "add a test for parse_config" --json | tail -1 | jq .
```

A failing run looks like this — also a checked-in fixture
(`tests/protocol_fixtures/tool_failure.jsonl`):

```jsonl
{"type":"tool_result","name":"bash","status":"error"}
{"type":"result","status":"incomplete","exit_code":1,"reason":"truncated_tool_call_exhausted","tokens":{"prompt":3,"completion":3,"total":6}}
```

## Not the same as the project event log

`<project>/.localcode/events.jsonl` is a separate, always-on, append-only local
audit trail (server lifecycle, tool calls, redactions, turn boundaries). It is
written whether or not you use `--json`, and it never leaves the machine.
