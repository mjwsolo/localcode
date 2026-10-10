#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
printf '$29\nZX-4471\n' > answer.txt
mkdir -p site && printf 'Eval Bot\n' > site/contact.log
