# Demo recordings

The GIFs under `website/public/demo/` (and the README hero copy under
`docs/assets/demo/`) are recorded from the real app with
[vhs](https://github.com/charmbracelet/vhs). Re-record after any change to
the first-run flow:

    brew install vhs gifsicle
    # a project with a failing test, and the model already downloaded
    cd website/demo-tapes && ./record.sh

Each `.tape` drives localcode 0.4.2 in `/private/tmp/retry-demo`. `record.sh`
captures PNG frames with VHS, builds crisp GIFs with ffmpeg, checks
that pytest passes after the full-turn take, and copies the assets into
place. The shorter verification GIF comes from the edit-and-test frames of
that same take. The full terminal, including its status line, stays in frame.

The model must already be downloaded. The recorder links downloaded GGUFs into
`/private/tmp/localcode-models` so the picker does not show a personal path.
If a model server is already running, it must use that temporary models folder
and have `Qwen3.6-35B-A3B-UD-Q8_K_XL` loaded; the script checks both before
recording. Restore your usual model folder afterward.
