# Demo recordings

The GIFs under `docs/public/demo/` (and the README hero copy under
`docs/assets/demo/`) are recorded from the real app with
[vhs](https://github.com/charmbracelet/vhs). Re-record after any change to
the first-run flow:

    brew install vhs ffmpeg gifsicle
    uv tool install localcode
    cd docs/demo-tapes && ./record.sh            # all tapes
    ./record.sh first-change                      # one tape

## The demo task

Every tape runs against a synthetic project made by `make_demo_repo.py`: a
TypeScript web app with 678 commits over a year by five fictional
contributors. The task is "Git Wrapped": LocalCode builds a Spotify-Wrapped
style page for that history. It writes `wrapped.py` (reads `git log`, writes
`wrapped.json`) and a self-contained `wrapped.html` of animated full-screen
cards, runs the script, then opens the page. The prompt is the same in all
tapes. `record.sh` rebuilds the repo before each take.

## Tapes

| Tape | Shows |
| --- | --- |
| `first-change.tape` | README hero: launch, prompt, agent working, result |
| `step-2-choose-model.tape` | the `/models` picker, model list then quants |
| `step-3-ask.tape` | typing and sending the prompt |
| `step-4-verify.tape` | the end of the run, files written and page opened |

`Hide`/`Show` cuts out the long waits. Each timing that depends on the model's
speed carries a `# TUNE` comment. Do a dry run, check where the agent actually
is at each `Show`, and adjust those sleeps.

## Pipeline

`record.sh` captures PNG frames with VHS, builds crisp GIFs with ffmpeg
(palette per take), shrinks them with gifsicle when installed, checks that
`wrapped.py`, `wrapped.json` and `wrapped.html` exist after the full-run
takes, and copies the assets into place. The full terminal, including its
status line, stays in frame at 1280x800.

## Model

The model must already be downloaded. The recorder expects
`Qwen3.6-35B-A3B-UD-Q8_K_XL` and links downloaded GGUFs into
`/private/tmp/localcode-models` so the picker does not show a personal path.
If a model server is already running, it must use that temporary models folder
and have that model loaded; the script checks both before recording. Restore
your usual model folder afterward. The agent opens `wrapped.html` in the
default browser at the end of a run, so expect a browser window to appear.
