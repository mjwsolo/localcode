---
name: git-commit-safely
description: Stage and commit safely — new commit not amend, no --no-verify, no `git add -A`, match project message style.
when_to_use: User asks to commit. ALSO when they say "ship" / "save progress" / "wrap this up".
---

Create a git commit safely.

Pre-commit checks (run in parallel):
- `git status` — see everything that's changed. NEVER use `-uall` (memory issue on large repos).
- `git diff --cached` — what's already staged.
- `git log --oneline -5` — the project's message style. Match it.

Rules:
- **NEW commits only.** Don't `--amend` unless the user explicitly says "amend". A failed pre-commit hook means the commit did NOT happen — amending would rewrite the PREVIOUS commit and potentially destroy work.
- **Never `git add -A` or `git add .`** — stage specific paths by name. Prevents accidentally committing `.env`, credentials, build output.
- **Never skip hooks** (`--no-verify`, `--no-gpg-sign`). Fix the hook failure; don't bypass.
- **Never force-push to main/master.** Warn before any force-push.

Use a HEREDOC for multi-line messages:
```
git commit -m "$(cat <<'EOF'
Summary line here.

Optional body explaining why.
EOF
)"
```

If there's nothing staged, don't create an empty commit. Tell the user.
