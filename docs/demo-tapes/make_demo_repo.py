"""Create the demo project: a small web app with a year of synthetic git history (fictional contributors)."""
import os, random, subprocess, sys, datetime, shutil
from pathlib import Path
random.seed(7)
root = Path(sys.argv[1]); shutil.rmtree(root, ignore_errors=True); root.mkdir(parents=True)
authors = [("Alex Rivera", "alex@example.dev", 0.38), ("Sam Okafor", "sam@example.dev", 0.27), ("Jordan Lee", "jordan@example.dev", 0.18), ("Priya Nair", "priya@example.dev", 0.12), ("Max Weber", "max@example.dev", 0.05)]
files = ["src/app.ts", "src/router.ts", "src/api/client.ts", "src/api/auth.ts", "src/ui/Button.tsx", "src/ui/Modal.tsx", "src/ui/theme.css", "src/store.ts", "README.md", "package.json", "tests/router.test.ts", "tests/store.test.ts", "docs/setup.md"]
weights = [9, 6, 5, 4, 5, 3, 4, 7, 2, 2, 3, 3, 1]
msgs = ["fix flaky test", "add dark mode toggle", "refactor router", "bump deps", "handle 401 refresh", "polish modal animation", "typo", "speed up store selectors", "add retry to api client", "wip", "docs: setup steps", "fix mobile layout", "ship v{v}", "revert accidental console.log", "add keyboard shortcuts", "cache user profile", "lint", "rename store slices"]
def git(*a, env=None): subprocess.run(["git", *a], cwd=root, check=True, env={**os.environ, **(env or {})}, stdout=subprocess.DEVNULL)
git("init", "-q", "-b", "main")
for f in files: (root / f).parent.mkdir(parents=True, exist_ok=True); (root / f).write_text(f"// {f}\n")
start = datetime.datetime(2025, 10, 27, 0, 0); v = 1
day = start
while day < datetime.datetime(2026, 10, 24):
    busy = day.weekday() < 5 and random.random() < 0.82
    n = random.choice([1, 1, 2, 3, 4, 6]) if busy else (1 if random.random() < 0.12 else 0)
    if day.month == 3 and 10 <= day.day <= 24: n += 3  # a launch crunch
    for _ in range(n):
        a = random.choices(authors, [w for *_, w in authors])[0]
        t = day + datetime.timedelta(hours=random.choices([9, 10, 11, 14, 15, 16, 17, 21, 22, 23], [3, 5, 5, 4, 6, 5, 3, 3, 6, 4])[0], minutes=random.randint(0, 59))
        f = random.choices(files, weights)[0]
        with open(root / f, "a") as h: h.write(f"// change {t.isoformat()}\n")
        m = random.choice(msgs).format(v=f"0.{v}"); v += m.startswith("ship")
        env = {"GIT_AUTHOR_NAME": a[0], "GIT_AUTHOR_EMAIL": a[1], "GIT_COMMITTER_NAME": a[0], "GIT_COMMITTER_EMAIL": a[1], "GIT_AUTHOR_DATE": t.isoformat(), "GIT_COMMITTER_DATE": t.isoformat()}
        git("add", "-A", env=env); git("commit", "-q", "-m", m, env=env)
    day += datetime.timedelta(days=1)
print(subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip(), "commits")
