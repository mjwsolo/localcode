#!/bin/bash
# Live check of the shell sandbox profile (macOS only, no GPU, no model).
#
#   scripts/sandbox-matrix.sh PROFILE.sb [nonet]
#
# PROFILE.sb is the profile the runtime generates (ShellSandbox.profile(true) in the UI fork's
# packages/core/src/shell/sandbox.ts; profile(false) with `nonet`). Every case runs under
# `sandbox-exec` inside a scratch project. Build tools use normal caches. Home and credential probes use fixtures.
set -eu
PROFILE="${1:?profile file}"
MODE="${2:-net}"
[ "$(uname)" = Darwin ] || { echo "macOS only"; exit 0; }

ROOT="$(mktemp -d "${HOME:?}/.sbmatrix.XXXXXX")" || exit 1
ROOT="$(cd -- "$ROOT" && pwd -P)" || exit 1
readonly ROOT
trap 'rm -rf -- "$ROOT"' EXIT
PROJECT="$ROOT/my project (1)"
FAKEHOME="$ROOT/home"
mkdir -p "$PROJECT" "$FAKEHOME/.ssh" "$FAKEHOME/.aws"
echo "PRIVATE KEY" > "$FAKEHOME/.ssh/id_rsa"
echo "aws_secret=1" > "$FAKEHOME/.aws/credentials"
# Tool state lives under the real HOME for read-only reuse; keep the real HOME for tool caches.
REALHOME="$HOME"
TMPD="$(cd "${TMPDIR:-/tmp}" && pwd -P)"

pass=0; fail=0
sb() { # run a snippet in the sandbox, cwd = project
  /usr/bin/sandbox-exec -f "$PROFILE" -D "PROJECT=$PROJECT" -D "HOME=$REALHOME" -D "TMPDIR=$TMPD" \
    /bin/zsh -fc 'cd -- "$1" && eval "$2"' sb "$PROJECT" "$1" 2>&1
}
sbfake() { # same, but HOME names the fake home so the secret rules point at the fake secrets
  HOME="$FAKEHOME" /usr/bin/sandbox-exec -f "$PROFILE" -D "PROJECT=$PROJECT" -D "HOME=$FAKEHOME" -D "TMPDIR=$TMPD" \
    /bin/zsh -fc 'cd -- "$1" && eval "$2"' sb "$PROJECT" "$1" 2>&1
}
ok() { # name, command: must exit 0
  local out rc=0; out="$(sb "$2")" || rc=$?
  if [ $rc -eq 0 ]; then echo "PASS  $1"; pass=$((pass+1)); else echo "FAIL  $1 (exit $rc)"; echo "$out" | tail -4 | sed 's/^/        /'; fail=$((fail+1)); fi
}
blocked() { # name, runner, command: must exit nonzero
  local out rc=0; out="$("$2" "$3")" || rc=$?
  if [ $rc -ne 0 ]; then echo "PASS  $1 (blocked)"; pass=$((pass+1)); else echo "FAIL  $1 (was NOT blocked)"; echo "$out" | tail -3 | sed 's/^/        /'; fail=$((fail+1)); fi
}

# A launcher failure must never count as a successful policy denial.
sb 'true' || exit 1
sbfake 'true' || exit 1

if [ "$MODE" = nonet ]; then
  ok "localhost server + curl (network off)" 'python3 -m http.server 18765 --bind 127.0.0.1 >/dev/null 2>&1 & p=$!; sleep 1.5; curl -sf http://127.0.0.1:18765/ >/dev/null; r=$?; kill $p; exit $r'
  blocked "outbound internet (network off)" sb 'curl -sf -m 5 https://registry.npmjs.org/ >/dev/null'
else
  ok "git init, commit, status" 'git init -q && git config user.email a@b.c && git config user.name t && echo x > f && git add f && git commit -qm init && git status --short'
  if command -v npm >/dev/null; then
    ok "npm init + install tiny package (network)" 'npm init -y >/dev/null && npm install --no-audit --no-fund is-odd >/dev/null'
  fi
  if command -v uv >/dev/null; then
    ok "uv venv + pip install (network)" 'uv venv -q .venv && uv pip install -q --python .venv/bin/python six'
  else
    ok "python venv + pip install (network)" 'python3 -m venv .venv && .venv/bin/pip install -q six'
  fi
  if command -v cargo >/dev/null; then
    ok "cargo new + build" 'cargo new -q hello_c && cd hello_c && cargo build -q'
  else
    echo "SKIP  cargo (not installed)"
  fi
  ok "make" 'printf "all:\n\t@echo built > out.txt\n" > Makefile && make -s && test -s out.txt'
  ok "clang hello" 'printf "#include <stdio.h>\nint main(){puts(\"hi\");return 0;}\n" > h.c && clang h.c -o h && ./h'
  ok "python http.server + curl localhost" 'python3 -m http.server 18766 --bind 127.0.0.1 >/dev/null 2>&1 & p=$!; sleep 1.5; curl -sf http://127.0.0.1:18766/ >/dev/null; r=$?; kill $p; exit $r'
  if command -v bun >/dev/null; then
    ok "bun test" 'printf "import {test,expect} from \"bun:test\"\ntest(\"a\",()=>expect(1).toBe(1))\n" > a.test.ts && bun test a.test.ts'
  fi
  ok "write to temp dir" 'echo t > "$TMPDIR/sbm-$$" && rm "$TMPDIR/sbm-$$"'
fi

# Repository metadata: hooks run later outside the sandbox, so they are read-only here. .git/config stays writable
# (git init, branch, remote and fetch write it).
ok "git branch, checkout, remote add, config write" 'git init -q 2>/dev/null; git -c user.email=a@b.c -c user.name=t commit -q --allow-empty -m a; git branch b && git checkout -q b && git remote add o https://example.invalid/x.git && git config user.name x'
blocked "write .git/hooks/pre-commit" sb 'git init -q 2>/dev/null; printf "#!/bin/sh\nexit 0\n" > .git/hooks/pre-commit'
blocked "write project localcode.json" sb 'echo {} > localcode.json'
blocked "write project .mcp.json" sb 'echo {} > .mcp.json'
blocked "write project .localcode-agent/x" sb 'mkdir -p .localcode-agent && echo x > .localcode-agent/x'
blocked "write ~/.zshrc (fake home)" sbfake 'echo x >> "$HOME/.zshrc"'
blocked "write ~/.ZSHRC, other case (fake home)" sbfake 'echo x >> "$HOME/.ZSHRC"'
blocked "write .GIT/hooks, other case" sb 'git init -q 2>/dev/null; mkdir -p .GIT/hooks && printf "#!/bin/sh\n" > .GIT/hooks/pre-commit'
blocked "rename .git to plant a hook" sb 'git init -q 2>/dev/null; mv .git .g && mkdir -p .g/hooks && printf "#!/bin/sh\n" > .g/hooks/pre-commit && mv .g .git'
blocked "reach the Docker socket" sb 'test -S /var/run/docker.sock || exit 1; curl -sf --unix-socket /var/run/docker.sock http://x/version'
blocked "write ~/.config/localcode-agent (fake home)" sbfake 'mkdir -p "$HOME/.config/localcode-agent" && echo x > "$HOME/.config/localcode-agent/cli.json"'
blocked "write to ~/sandbox-test-x" sbfake 'echo x > "$HOME/sandbox-test-x"'
blocked "write outside project via cd .." sb 'cd .. && echo x > sbm-escape.txt'
blocked "script that cd's out and writes" sbfake 'printf "cd /\ncd \"$HOME\"\necho x > sbm-script-out\n" > s.sh && sh s.sh'
blocked "python reads ~/.ssh (fake home)" sbfake 'python3 -c "import os; open(os.path.expanduser(\"~/.ssh/id_rsa\")).read()"'
blocked "cat ~/.aws/credentials (fake home)" sbfake 'cat "$HOME/.aws/credentials"'
blocked "ls ~/.ssh (fake home)" sbfake 'ls "$HOME/.ssh"'
# Toolchains live anywhere in HOME (conda, uv-managed Pythons, a virtualenv in another folder).
# An allow-list of home folders once made such a Python crash: "No module named 'encodings'".
OTHER="$ROOT/other project"; mkdir -p "$OTHER"; /usr/bin/python3 -m venv "$OTHER/.venv" >/dev/null 2>&1
echo "notes" > "$OTHER/readme.txt"
ok "python from a virtualenv in another home folder" "\"$OTHER/.venv/bin/python\" -c 'import encodings, json, zipfile; print(1)'"
ok "read a file in another home folder" "cat \"$OTHER/readme.txt\""
blocked "write a file in another home folder" sb "echo x > \"$OTHER/written.txt\""
mkdir -p "$FAKEHOME/Library/Messages" && echo "chat" > "$FAKEHOME/Library/Messages/chat.db"
blocked "read Messages (fake home)" sbfake 'cat "$HOME/Library/Messages/chat.db"'
[ -e "$FAKEHOME/sandbox-test-x" ] && { echo "FAIL  ~/sandbox-test-x exists"; fail=$((fail+1)); }
[ -e "$ROOT/sbm-escape.txt" ] && { echo "FAIL  escape file written"; fail=$((fail+1)); }

echo "passed $pass, failed $fail"
[ $fail -eq 0 ]
