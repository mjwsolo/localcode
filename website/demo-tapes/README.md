# Demo recordings

The GIFs under `website/public/demo/` (and the README hero copy under
`docs/assets/demo/`) are recorded from the real app with
[vhs](https://github.com/charmbracelet/vhs). Re-record after any change to
the first-run flow:

    brew install vhs gifsicle
    # a project with a failing test, and the model already downloaded
    cd website/demo-tapes && ./record.sh

Each `.tape` drives `localcode` in a scratch project; `record.sh` prepares the
project, runs the tapes, optimises the GIFs and copies them into place.
