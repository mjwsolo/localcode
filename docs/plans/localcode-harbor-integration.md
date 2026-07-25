# LocalCode Harbor Integration Plan

Status: proposed for review  
Date: 2026-07-25  
Implementation status: no code written

## 1. Decision summary

Build a separately installable **Harbor custom-agent plugin** inside the
LocalCode repository, initially under `integrations/harbor/`.

The plugin owns the evaluation lifecycle, but runs `localcode run --json` as a
supervised child process inside Harbor's task environment. It does not import
and execute `LocalCodeApp` inside Harbor's Python process.

The first release supports one model topology:

1. Harbor and the LocalCode agent run inside an isolated Linux task container.
2. A single model server runs natively on the Mac and uses Metal.
3. LocalCode connects to that server through an explicitly configured endpoint.
4. The plugin captures LocalCode's JSONL events and produces a Harbor trajectory.
5. Harbor's verifier, outside LocalCode, determines whether the task passed.

This gives us realistic LocalCode behavior, task isolation, one model in memory,
and independent grading.

## 2. Why we are doing this

LocalCode currently has unit and integration tests, but they cannot answer the
product-level questions:

- Can a particular small local model finish realistic coding tasks?
- Did an agent-loop change improve completion rate or merely move failures?
- How much wall time, memory, context, and generation does each success cost?
- Does LocalCode regress relative to Codex CLI, Claude Code, OpenCode, or Pi on
  the same task and verifier?
- Which failures come from the model, the model server, LocalCode's policy, or
  the execution environment?

Harbor already supplies task isolation, datasets, timeouts, verifiers, result
storage, and comparable agent runs. The missing component is a faithful
LocalCode integration. Building another benchmark runner would duplicate the
least product-specific part of the system.

## 2.1 Relationship to LocalCode's existing evaluation harness

The `eval/trustworthy-harness-and-matrix` branch already contains a substantial
LocalCode-native evaluation system:

- a headless runner and model/config matrix;
- 14 fast, deterministic LocalCode tasks;
- Aider Polyglot Python and JavaScript adapters;
- local SWE-bench Verified and LiveCodeBench adapters;
- partial-credit verification;
- pass@k and pass^k repeatability measurements;
- JSONL transcript retention;
- model-based trajectory judging;
- a regression/failure flywheel.

That work is not replaced by Harbor. The two systems serve different feedback
loops:

| Need | Local `eval/` harness | Harbor |
|---|---:|---:|
| Fast iteration on a developer's Mac | Primary | Too heavy |
| Native Metal process/resource measurements | Primary | Partial |
| No-Docker operation | Yes | No for Terminal-Bench |
| LocalCode-specific regression tasks | Primary | Optional export |
| Standard container-isolated tasks | Limited | Primary |
| Terminal-Bench 2.0 | No | Primary |
| Same harness as Codex/Claude Code/OpenCode/Pi | No | Primary |
| Cloud or large parallel evaluation | No | Primary |
| Independent standardized result/trajectory format | Partial | Primary |

The architecture is therefore **two tiers, one protocol**:

```text
                         versioned LocalCode JSONL
                                    |
                  +-----------------+-----------------+
                  |                                   |
        native eval/ harness                 Harbor agent plugin
        fast, Mac, every change              isolated, comparative, periodic
                  |                                   |
       LocalCode regression suite          TB2 + other Harbor datasets
```

Rules preventing duplicated infrastructure:

1. `localcode run --json` is the sole machine-facing LocalCode execution
   protocol for both harnesses.
2. Event parsing and failure normalization **must** live in a small public
   LocalCode protocol module. Both consumers import it; neither harness may
   maintain its own interpretation of LocalCode events.
3. The native `eval/` tasks remain the source of truth for LocalCode-specific
   regressions.
4. If those tasks need container isolation, write a mechanical exporter into
   Harbor task format. Do not manually duplicate task instructions or graders.
5. Terminal-Bench and other public Harbor datasets remain external and
   unmodified.
6. Harbor is run periodically and before consequential releases. The native
   harness remains the short feedback loop used during development.

Before implementing this plan, reconcile and land the evaluation branch or
explicitly rebase the integration work on top of it. Do not build a second
runner on the main branch while an existing runner waits unaccounted for on a
feature branch.

## 3. Terminology

Harbor uses two related extension types:

- A **custom agent** tells Harbor how to install and run an agent such as
  LocalCode.
- A benchmark **adapter** converts another dataset into Harbor's task format.

LocalCode needs a custom installed agent, not a dataset adapter. This document
calls the distributable package a "plugin" from the user's perspective and its
main Python class the "Harbor agent integration."

## 4. Why the plugin launches a process

The process launch happens *inside the plugin*:

```text
harbor run
    |
    +-- loads LocalCodeAgent plugin
            |
            +-- installs/configures LocalCode in task container
            |
            +-- starts: localcode run --goal ... --json
            |                |
            |                +-- connects to host model server
            |                +-- reads/edits/runs inside task container
            |
            +-- parses JSONL and records Harbor trajectory
```

The child-process boundary is intentional:

1. **It tests the shipped product.** Users invoke the LocalCode CLI. Importing
   internal classes would test a private integration path with different
   initialization, signals, output handling, configuration, and cleanup.
2. **It preserves task isolation.** LocalCode's shell and file tools operate
   inside the Harbor container. Importing LocalCode into the host Harbor process
   would either let tools touch the host or require a second remote-tool
   abstraction that LocalCode users do not exercise.
3. **It gives Harbor control.** Harbor can apply timeouts, terminate the entire
   process tree, capture stdout/stderr, and preserve partial output after a
   crash.
4. **It avoids Python-runtime coupling.** Harbor and LocalCode can use different
   dependency versions without sharing global state, logging, signal handlers,
   event loops, or module singletons.
5. **It supports version comparisons.** The same plugin can install a released
   LocalCode version, a wheel, or the checked-out source.

An in-process mode is explicitly rejected for v1. It saves little startup time
relative to model inference and creates a second, less realistic execution
path. We can reconsider it only if profiling shows process startup materially
affects short-task measurements.

## 5. Repository and distribution design

Keep the plugin in the LocalCode repository at first:

```text
integrations/harbor/
├── pyproject.toml
├── README.md
├── src/localcode_harbor/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── events.py
│   ├── trajectory.py
│   └── telemetry.py
└── tests/
    ├── fixtures/
    ├── test_agent.py
    ├── test_config.py
    ├── test_events.py
    ├── test_trajectory.py
    └── test_telemetry.py
```

Reasons:

- Agent changes and event-schema changes remain reviewable in one pull request.
- CI can test LocalCode and the integration together.
- We avoid a second repository, release process, issue tracker, and version-skew
  problem before the interface is stable.
- It remains a distinct Python distribution, so Harbor users do not need to
  install benchmark dependencies with normal LocalCode.

After the contract survives at least two LocalCode releases, submit a thin
first-class `localcode` agent to Harbor upstream. The source of truth should
remain the smallest possible package; do not maintain two divergent adapters.

Proposed install during development:

```bash
uv tool install harbor
uv pip install -e ./integrations/harbor
harbor run \
  --dataset terminal-bench@2.0 \
  --agent localcode_harbor.agent:LocalCodeAgent \
  --model local/<profile> \
  --n-concurrent 1
```

The exact `--model` value will be finalized after testing Harbor's model field
validation. LocalCode's model endpoint and runtime settings are agent
configuration, not Harbor provider credentials.

## 6. Model-server architecture for 16 GB Macs

Do not put one model inside every task container and do not run concurrent
models. Keep one native server:

```text
macOS host
├── llama.cpp/MLX server
│   ├── Metal GPU
│   ├── one loaded model
│   └── bounded KV cache
├── Harbor coordinator
└── Docker task container (concurrency = 1)
    └── LocalCode CLI
        └── HTTP -> host.docker.internal:<port>
```

Defaults:

- `--n-concurrent 1`
- one warm model server reused across trials
- no speculative draft model
- no model download during a measured trial
- per-run isolated `LOCALCODE_HOME`
- task timeout enforced by Harbor and LocalCode

We will report cold-start and warm-run measurements separately. Model download
time is setup cost, not agent performance.

## 7. Required LocalCode core contract

The plugin must use public CLI behavior. Add only the capabilities it cannot
obtain today.

### 7.1 Existing-server mode

Current `run_headless_json()` resolves a local GGUF and starts or restarts a
server. Add an explicit headless option:

```bash
localcode run \
  --goal "..." \
  --json \
  --base-url http://host.docker.internal:8081 \
  --server-mode external
```

Semantics:

- `managed`: current behavior; LocalCode may start/restart its server.
- `external`: require a healthy configured endpoint; never start, restart,
  download, or terminate the server.

Failure must produce a final JSON result with a stable machine-readable reason,
such as `model_server_unreachable`, not just free-form exception text.

Environment equivalents should exist for container configuration:

```text
LOCALCODE_BASE_URL
LOCALCODE_SERVER_MODE=external
```

### 7.2 Versioned JSONL event contract

Add a schema version to every record or to an initial metadata record:

```json
{"type":"run_start","schema_version":1,"run_id":"...","localcode_version":"..."}
```

The terminal `result` event remains mandatory even after timeout, interruption,
or agent error where LocalCode still controls shutdown. Required terminal
fields:

- status and stable reason code
- LocalCode version and run ID
- elapsed time
- prompt, completion, and total tokens when provided by the backend
- rounds and tool-call counts
- final response

The plugin must tolerate unknown event types and fields so LocalCode can extend
the schema without breaking old plugins.

### 7.2.1 One public parser and normalizer

Create a dependency-light public module owned by LocalCode, proposed as:

```text
src/localcode/protocol/
├── __init__.py
├── events.py       # typed event envelope and schema-version handling
├── jsonl.py        # incremental stream parser
└── outcomes.py     # stable reason/failure normalization
```

Dependency direction:

```text
LocalCode JSON emitter
          |
          v
localcode.protocol  <----- native eval/ runner
          ^
          |
    Harbor plugin
```

The protocol package must:

- use only Python's standard library unless an existing lightweight LocalCode
  dependency is unavoidable;
- parse incrementally from text or byte streams;
- preserve unknown event types and fields;
- distinguish malformed input, truncated input, unsupported major schema, and
  missing/duplicate terminal events;
- expose typed normalized summaries without discarding raw records;
- normalize stable LocalCode exit reasons independently from benchmark pass or
  failure;
- redact configured secret fields before persistence;
- support a compatibility policy where readers accept the same major schema
  and ignore unknown additive fields.

The native eval branch's `parse_jsonl_metrics()` must be replaced with this
module, not retained as a second parser. The Harbor plugin's `EventParser`
becomes a thin streaming consumer of the same module rather than an independent
implementation.

This parser is part of LocalCode's public automation API. Its compatibility is
tested and versioned with the JSONL schema.

### 7.3 Deterministic benchmark configuration

Allow all material runtime choices to be passed without reading a user's normal
configuration. Each trial gets a temporary `LOCALCODE_HOME` containing a
generated, archived configuration. Record:

- model/profile and quantization
- context and generation limits
- thinking mode and budget
- sampler settings
- tool policy and maximum rounds
- LocalCode, plugin, server, and model identifiers
- seed when the backend supports one

Secrets must never appear in the archived configuration or trajectory.

### 7.4 Process termination

On timeout or interruption:

1. ask the LocalCode process group to terminate;
2. allow a short grace period;
3. kill the remaining process group;
4. preserve stdout, stderr, events, and partial trajectory;
5. do not terminate an external model server.

## 8. Plugin components

### `LocalCodeAgent`

Subclass Harbor's `BaseInstalledAgent`.

Responsibilities:

- install the selected LocalCode artifact in the task container;
- validate that the external model endpoint is reachable;
- create an isolated LocalCode configuration;
- quote the task instruction safely;
- run the CLI through Harbor's `exec_as_agent`;
- populate Harbor's context after normal exit, timeout, or failure.

It must not grade the task or modify verifier output.

### `LocalCodeConfig`

A validated immutable configuration object covering:

- install source: release version, wheel URL/path, or editable source;
- CLI command and timeout;
- model endpoint and identifier;
- LocalCode runtime overrides;
- telemetry level;
- allowed environment-variable names.

Reject ambiguous combinations, especially `server_mode=external` without a
base URL and concurrent trials against a memory-constrained local server.

### `EventParser`

Thinly integrate Harbor's process stream with `localcode.protocol`; do not
reimplement JSONL parsing or LocalCode outcome normalization. It incrementally
consumes JSONL without loading the full transcript into memory.

Rules:

- stdout is the protocol channel;
- stderr is diagnostic output;
- malformed lines are recorded and skipped up to a small threshold;
- unknown event types are retained but do not fail parsing;
- duplicate terminal events and missing terminal events are protocol errors;
- partial final lines after forced termination are retained as diagnostics.

### `TrajectoryBuilder`

Map LocalCode events to Harbor's Agent Trajectory Interchange Format where
possible:

- instruction and agent response messages;
- reasoning summaries only when LocalCode intentionally emits them;
- tool name, arguments, result, duration, and error status;
- token usage and timing;
- termination reason.

Never infer successful edits or commands from assistant prose. Only tool-result
events count as actions.

### `TelemetryCollector`

Two scopes must remain separate:

1. **Agent-container telemetry:** LocalCode process RSS/CPU, elapsed time, I/O,
   exit status.
2. **Host-model telemetry:** model-server RSS, host memory pressure, generation
   throughput, and optional macOS power/thermal counters.

Core correctness does not depend on privileged telemetry. `powermetrics` and
similar host counters are optional because requiring `sudo` would make the
basic benchmark difficult to run and unsafe to automate.

## 9. Data flow

```text
Terminal-Bench task
      |
      v
Harbor creates isolated container
      |
      v
LocalCodeAgent.install()
      |  installs pinned LocalCode artifact
      v
LocalCodeAgent.run(instruction)
      |  writes isolated config
      |  starts telemetry
      |  starts LocalCode CLI
      v
LocalCode JSONL stdout -----> EventParser -----> TrajectoryBuilder
      |                              |                   |
      |                              +---- diagnostics   +---- Harbor context
      v
tools modify task workspace
      |
      v
Harbor verifier inspects workspace
      |
      v
reward + trajectory + resource metrics + reproducibility manifest
```

The verifier is authoritative. A LocalCode `status=ok` means the process
completed its own loop, not that the benchmark task passed.

## 10. Failure taxonomy

Use stable categories so regressions can be compared:

- `setup_failure`
- `model_server_unreachable`
- `model_server_failure`
- `agent_timeout`
- `agent_crash`
- `agent_incomplete`
- `event_protocol_error`
- `tool_failure`
- `context_exhausted`
- `reasoning_loop_abort`
- `verifier_failure`
- `task_failed`
- `task_passed`

Preserve the raw LocalCode reason alongside the normalized category.

## 11. Security decisions

- Treat task instructions and repository contents as untrusted.
- Never construct the CLI command with unquoted task text. Prefer an
  instruction file or stdin if Harbor's execution API supports it reliably.
- Use an allowlist for environment variables copied into the container.
- Redact credentials and authorization headers from events and diagnostics.
- Do not mount the developer's normal LocalCode home directory.
- Do not give benchmark containers the Docker socket.
- Bind the host model server only as broadly as required and use an
  unguessable token if the server supports authentication.
- Keep Harbor's verifier independent from LocalCode and its generated files.

## 12. Testing plan

### Unit tests

- every supported JSONL event and unknown future events;
- chunked, truncated, malformed, and non-UTF-8 output;
- missing/duplicate terminal results;
- configuration validation and secret redaction;
- command/instruction quoting;
- failure-category normalization;
- trajectory mapping;
- process-group termination;
- telemetry unavailable and permission-denied paths.

### Contract tests

Run a fake `localcode` executable that emits controlled streams:

```text
normal success       malformed event       no terminal event
LocalCode error      timeout/partial line  huge tool result
signal termination  unknown schema field  stderr flood
```

Assert the Harbor context, artifacts, classification, and cleanup for each.

### LocalCode core tests

- external mode never starts, restarts, downloads, or kills a server;
- unreachable endpoint returns the stable JSON result;
- every controlled exit emits exactly one terminal result;
- schema-version compatibility;
- native `eval/` and Harbor fixtures produce identical normalized summaries
  through `localcode.protocol`;
- unknown additive fields remain available while older consumers continue;
- unsupported major schemas fail explicitly rather than being misinterpreted;
- isolated `LOCALCODE_HOME` ignores normal user configuration.

### End-to-end smoke tests

1. Harbor hello-world task with a fake deterministic model endpoint.
2. One small Terminal-Bench task against a real local model.
3. Forced timeout proving partial artifacts survive.
4. Two sequential trials proving the model stays loaded and task state does
   not leak.

Do not put a real-model test in the default CI suite. Mark it explicitly and
run it on a suitable self-hosted Mac or manually.

## 13. Performance measurement

Record at minimum:

- verifier reward/pass;
- total and per-round wall time;
- time to first token when exposed;
- prompt/completion tokens and tokens per second;
- number and duration of tool calls;
- LocalCode peak RSS;
- model-server peak RSS;
- host memory-pressure state;
- timeout and failure category.

Compare configurations using repeated runs, not a single sample. Report pass
rate and distributions. Do not optimize tokens-per-second at the expense of
task completion.

## 14. Implementation phases

### Phase 0: compatibility and topology spike

- Pin a Harbor version.
- Implement a throwaway custom `BaseInstalledAgent` invoking a fake LocalCode.
- Verify custom-agent loading, task working directory, stdin/quoting behavior,
  timeout behavior, artifact locations, and trajectory APIs.
- Run Harbor's hello-world task.
- Start a trivial HTTP server on macOS bound exactly as the future model server
  will be bound.
- From Harbor's Linux task container, prove access through
  `host.docker.internal`, including request/response streaming.
- Record the required bind address, Docker Desktop setting, firewall behavior,
  hostname resolution, and failure message when the host is unreachable.
- Verify Harbor's supported container architecture and image behavior on Apple
  Silicon.

Exit criterion: we know the exact Harbor interfaces and have proved the
Docker-to-macOS-host network path without changing LocalCode or loading a
model.

#### Mandatory Phase 0 go/no-go

Proceed only if all of these are true:

- a custom installed agent loads without patching Harbor;
- LocalCode can run as the task user in the correct workspace;
- Harbor can terminate the agent and preserve partial artifacts;
- trajectory/context APIs can represent LocalCode's essential events;
- a Linux task container can reliably stream to the macOS host endpoint;
- the design does not require privileged Docker mounts or a second loaded
  model.

Stop the Harbor work if any failed condition requires a Harbor fork, a new
remote-tool layer, weakening task isolation, or ongoing platform-specific
patches. Keep Phase 1 only where it independently improves LocalCode and keep
using the native `eval/` harness.

### Phase 1: stabilize LocalCode's public runner

- Add external-server mode.
- Add versioned JSONL start/result contracts and stable reason codes.
- Implement `localcode.protocol` as the only parser and outcome normalizer.
- Replace the eval branch's private `parse_jsonl_metrics()` with the shared
  protocol module.
- Add golden-stream compatibility fixtures consumed by LocalCode core tests,
  the native eval harness, and later the Harbor plugin.
- Add deterministic isolated configuration.
- Add core contract tests and documentation.

Exit criterion: a shell script can run LocalCode repeatably against an existing
server without Harbor, and the native eval harness consumes the public protocol
without a private JSONL parser or private failure taxonomy.

### Phase 2: implement the plugin

- Create the nested package.
- Implement configuration, installed agent, parser, trajectory mapping, and
  basic container telemetry.
- Add unit and fake-process contract tests.

Exit criterion: Harbor hello-world succeeds and all simulated failures produce
useful artifacts.

### Phase 3: real local-model validation

- Document the native Metal model-server setup.
- Connect the task container through `host.docker.internal`.
- Run a small curated task set sequentially.
- Validate isolation, memory use, timeout cleanup, and warm-server reuse.

Exit criterion: repeatable results on a 16 GB Mac without loading a second
model.

### Phase 4: LocalCode regression suite

- Reuse the existing native suite on
  `eval/trustworthy-harness-and-matrix`; do not create another 10-20 task set.
- Audit its coverage of read, edit, test, debugging, tool recovery, context
  pressure, and loop prevention, then add only demonstrated gaps.
- Keep Terminal-Bench tasks unmodified for external comparability.
- If useful, mechanically export selected LocalCode tasks into Harbor format.
- Pin model, quant, server, sampler, LocalCode version, task revision, and
  machine class for every baseline.
- Run repeated trials and measure natural variance before defining regression
  thresholds.
- Report deltas advisory-only in v1. Do not block merges on stochastic
  real-model outcomes.

Exit criterion: the suite detects seeded or previously observed regressions,
and its repeat-run variance is characterized. Merge gating is a later,
separately approved decision after a sustained low false-alarm rate.

### Phase 5: distribution and upstreaming

- Build and test the plugin distribution in CI.
- Publish a version compatible with each supported LocalCode event schema.
- Propose a thin first-class LocalCode agent to Harbor once the API is stable.
- Document upgrades and compatibility policy.

Exit criterion: a user can install and run the integration without cloning the
LocalCode repository.

## 15. Proposed change surface

Expected LocalCode core changes:

- `src/localcode/entrypoint.py`: external-server CLI option.
- `src/localcode/headless_json.py`: external mode and event contract.
- `src/localcode/config.py`: server ownership/config support.
- `src/localcode/protocol/`: public incremental parser, event envelope, and
  outcome normalization.
- `eval/runner.py` after the eval branch is reconciled: consume the public
  protocol and remove `parse_jsonl_metrics()`.
- focused tests for these behaviors.
- CLI/headless documentation.

Expected integration changes:

- one nested package under `integrations/harbor/`;
- its tests and README;
- CI job that installs the package and runs fake-agent contract tests.

If implementation requires broad changes to the agent loop, tool dispatch, or
TUI, stop and reassess. The integration should consume the public runner, not
reshape LocalCode around Harbor.

## 16. Explicitly deferred

- Running multiple local models or speculative decoding.
- Parallel real-model trials on memory-constrained machines.
- Training or fine-tuning models.
- A new general LocalCode plugin runtime. The current internal plugin
  registration is disabled, and restoring it is a separate product/security
  decision.
- Replacing Harbor's task, verifier, result, or viewer systems.
- Automatic privileged macOS power telemetry.
- Upstream Harbor inclusion before the contract proves stable.

## 17. Decisions requiring approval

1. **Process boundary:** approve CLI subprocess execution inside the Harbor task
   container; reject in-process `LocalCodeApp` integration for v1.
2. **Repository:** approve a distinct package inside the LocalCode repository;
   defer a separate repository.
3. **Model topology:** approve one native host model server with sequential
   container trials as the supported 16 GB configuration.
4. **Core API:** approve explicit `external` versus `managed` server ownership
   and a versioned JSONL contract.
5. **Scope:** approve correctness, trajectory, and basic resource telemetry for
   v1; keep privileged Mac GPU/power metrics optional.
6. **Upstream strategy:** approve custom import-path plugin first, then propose
   first-class Harbor support after two stable LocalCode releases.
7. **Evaluation strategy:** approve a two-tier system where the existing
   `eval/` harness is the fast native loop and Harbor is the periodic isolated
   comparison layer, both consuming the same versioned JSONL protocol.

## 18. Recommended first action

First reconcile the `eval/trustworthy-harness-and-matrix` branch into the
intended development base. Then do Phase 0 before changing LocalCode. It is the
cheapest way to invalidate our assumptions about Harbor's installed-agent API,
process control, working directory, trajectory format, Apple Silicon container
support, and Docker-to-native-server networking. The spike should be disposable
and take no dependency on a real model.

Only after that spike passes should we commit to the LocalCode external-server
and JSON event changes in Phase 1.
