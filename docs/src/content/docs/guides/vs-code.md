---
title: VS Code and other editors
description: Run LocalCode in an editor's terminal, or point editor extensions at its local API.
---

LocalCode does not have an editor extension yet. There are two ways to use it
alongside VS Code or a similar editor today.

## Run LocalCode in the editor's terminal

LocalCode runs in any terminal, including VS Code's built-in one.

1. Open your project in VS Code.
2. Open the terminal (`` Ctrl+` ``) and run `localcode`. It starts in the
   terminal's current directory, which is the project folder.
3. Ask for changes as usual. Edited files update in the editor as LocalCode
   writes them.

### Review and undo changes

Use VS Code's **Source Control** view to see what LocalCode changed. Each edited
file shows a diff against the last commit. To undo, discard a file's changes in
Source Control, or run `git restore <file>` or `git stash` in the terminal.
Commit before a large task so there is always a clean point to return to.

If the project is not a git repository, run `git init` and make a first commit
before you start, so the diff view has something to compare against.

## Use LocalCode's model from an editor extension

While LocalCode is open, its model is available as an OpenAI compatible API on
your Mac. Extensions that accept a custom OpenAI endpoint can use it for chat
and edits. Run:

```bash
localcode api
```

It prints the base URL (for example `http://127.0.0.1:8123/v1`) and the model
name. Add `--show-key` to include the API key, or `--json` for machine readable
output. See [Local API](/localcode/guides/local-api) for details.

The key changes every time LocalCode starts, so extension settings need
updating after a restart. You can fix the port with `localcode api --port 9234`
(and `localcode api --port auto` to undo it), but not the key.

### Continue

In Continue's `config.yaml`, add a model with the `openai` provider:

```yaml
models:
  - name: LocalCode
    provider: openai
    model: <model name from localcode api>
    apiBase: http://127.0.0.1:8123/v1
    apiKey: <key from localcode api --show-key>
```

### Cline

In Cline's settings, set **API Provider** to **OpenAI Compatible**, then fill in:

- **Base URL**: the base URL from `localcode api`
- **API Key**: the key from `localcode api --show-key`
- **Model ID**: the model name from `localcode api`

Click **Verify** to check the connection.

Other extensions that offer an "OpenAI compatible" or "custom endpoint" option
use the same three values: base URL, API key and model name.

## Caveats

- The API serves the model only. The extension's own agent, not LocalCode's
  agent, decides what to do with it, so LocalCode's tools, permissions and
  completion checks do not apply.
- One model is loaded at a time, and it is shared with every LocalCode window.
  Closing the window that started the model server stops the endpoint.
- Local models are slower than hosted ones, and some extensions send very large
  prompts or expect tool calling features a model may not support. If an
  extension times out, raise its request timeout.
- The key is a secret. Do not commit it to a repository or paste it into a
  shared settings file.

## Troubleshooting

- **`localcode api` says there is no running model server.** Start `localcode`
  first and wait for the model to load.
- **Connection refused or 401 errors.** The port or key changed after a restart.
  Run `localcode api --show-key` again and update the extension.
- **The extension says the model is not found.** Use the exact model name that
  `localcode api` prints.
- **Responses are empty or very slow.** Check that the model is loaded (the dot
  beside **localcode** in the bottom bar is green) and that no other app is
  using the memory.

## What would you want from an extension?

We are collecting what people want before deciding whether to build one:
opening edited files, diffs in the editor, the current selection as context, or
something else. Tell us on
[issue #257](https://github.com/mjwsolo/localcode/issues/257).
