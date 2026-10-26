// Harness for the terminal screen snapshot tests: per session it starts the
// mock model/control server, writes a throwaway HOME and the product's own
// runtime config, then runs the real localcode-ui in a pseudo-terminal.
import { TuiTest } from "@microsoft/tui-test"
import { spawn, spawnSync } from "node:child_process"
import { existsSync, mkdirSync, readFileSync, rmSync, utimesSync, writeFileSync } from "node:fs"
import { createServer } from "node:net"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(HERE, "../..")
const UI_BIN = process.env.LOCALCODE_UI_BIN ?? join(REPO, "src/localcode/bin/localcode-ui")
const PYTHON = process.env.LOCALCODE_PY ?? "python3"
export const COLS = 120
export const ROWS = 36
export const UPDATE = process.argv.includes("--update-snapshots") || process.env.E2E_UPDATE === "1"

function freePort() {
  return new Promise((ok, fail) => {
    const s = createServer()
    s.on("error", fail)
    s.listen(0, "127.0.0.1", () => {
      const { port } = s.address()
      s.close(() => ok(port))
    })
  })
}

async function waitHttp(url, ms = 30_000) {
  const end = Date.now() + ms
  while (Date.now() < end) {
    try {
      if ((await fetch(url)).ok) return
    } catch {}
    await new Promise((r) => setTimeout(r, 100))
  }
  throw new Error(`not up: ${url}`)
}

// Text that changes between runs is replaced before comparing or snapshotting.
const MASKS = [
  [/\b\d+\.\d+\.\d+(?:a\d+|b\d+|rc\d+)?\b/g, "<version>"],
  [/Completed in \d+(?:\.\d+)?m?s/g, "Completed in <time>"],
  [/Ask anything… ".*?"/g, 'Ask anything… "<placeholder>"'],
  [/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]/g, "<spin>"],
]
export function clean(text) {
  let t = text.split("\n").map((l) => l.replace(/\s+$/, "")).join("\n").replace(/\n+$/, "")
  for (const [re, to] of MASKS) t = t.replace(re, to)
  return t
}

const SNAPDIR = join(HERE, "__snapshots__")

function lineDiff(want, got) {
  const a = want.split("\n"), b = got.split("\n"), out = []
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    if (a[i] !== b[i]) out.push(`line ${i + 1}\n  - ${JSON.stringify(a[i] ?? "")}\n  + ${JSON.stringify(b[i] ?? "")}`)
  }
  return out.slice(0, 12).join("\n")
}

/** Compare the settled screen with the committed text snapshot (or write it when updating). */
export function expectSnapshot(name, text) {
  const file = join(SNAPDIR, `${name}.txt`)
  const got = text + "\n"
  if (UPDATE || !existsSync(file)) {
    if (!UPDATE) throw new Error(`no snapshot ${name}; run scripts/ui_e2e.sh --update-snapshots`)
    mkdirSync(SNAPDIR, { recursive: true })
    writeFileSync(file, got)
    return
  }
  const want = readFileSync(file, "utf8")
  if (want !== got) {
    mkdirSync(join(HERE, "snapshot-diffs"), { recursive: true })
    writeFileSync(join(HERE, "snapshot-diffs", `${name}.actual.txt`), got)
    throw new Error(`snapshot ${name} differs (actual saved to tests/e2e/snapshot-diffs/${name}.actual.txt)\n${lineDiff(want, got)}`)
  }
}

/**
 * Open a session. opts.model false starts with no model loaded.
 * Returns { term, screen(), type(), key(), waitFor(), settle(), snap(), ws, close() }.
 */
export async function open(name, { model = true, files } = {}) {
  const root = join("/tmp", `lce-${process.pid}-${name.replace(/\W+/g, "_")}`)
  rmSync(root, { recursive: true, force: true })
  const home = join(root, "home")
  // Short and fixed so the footer path is identical everywhere.
  const ws = join("/tmp", `lce-ws-${name.replace(/\W+/g, "_")}`)
  rmSync(ws, { recursive: true, force: true })
  mkdirSync(home, { recursive: true })
  mkdirSync(ws, { recursive: true })
  for (const [f, text] of Object.entries(files ?? { "README.md": "# demo\n\nA tiny project.\n", "a.py": "def add(a, b):\n    return a + b\n" })) {
    writeFileSync(join(ws, f), text)
    const old = new Date(Date.now() - 3_600_000)
    utimesSync(join(ws, f), old, old)
  }
  // A file outside the workspace, for the permission journey.
  mkdirSync("/tmp/lce-outside", { recursive: true })
  writeFileSync("/tmp/lce-outside/notes.txt", "outside notes\n")
  spawnSync("git", ["init", "-q", "-b", "main"], { cwd: ws })

  const mockPort = await freePort()
  const mock = spawn(PYTHON, [join(HERE, "mock_server.py"), String(mockPort), ...(model ? [] : ["--no-model"])], { stdio: "ignore" })
  await waitHttp(`http://127.0.0.1:${mockPort}/health`)

  // The same runtime config a real session gets, written by the product code.
  const cfg = join(home, "session.json")
  const gen = spawnSync(
    PYTHON,
    ["-c", "import sys; from pathlib import Path; from localcode.ui.launch import write_config; write_config(Path(sys.argv[1]), port=int(sys.argv[2]), ctx=32768, alias=(sys.argv[3] or None))", cfg, String(mockPort), model ? "e2e-model" : ""],
    { env: { ...process.env, PYTHONPATH: join(REPO, "src") }, encoding: "utf8" },
  )
  if (gen.status !== 0) throw new Error(`write_config failed: ${gen.stderr}`)

  const env = {
    ...process.env,
    HOME: home,
    XDG_DATA_HOME: join(home, ".local/share"),
    XDG_CONFIG_HOME: join(home, ".config"),
    XDG_CACHE_HOME: join(home, ".cache"),
    XDG_STATE_HOME: join(home, ".local/state"),
    LOCALCODE_CONFIG: cfg,
    LOCALCODE_CONTROL_URL: `http://127.0.0.1:${mockPort}`,
    LOCALCODE_CONTROL_TOKEN: "e2e-token",
    LOCALCODE_TEST_HOME: home,
    LOCALCODE_AGENT_RUN_DIR: join(home, "run"),
    OPENCODE_DISABLE_LSP_DOWNLOAD: "1",
    OPENCODE_DISABLE_CLAUDE_CODE_SKILLS: "1",
    LOCALCODE_DISABLE_PROJECT_CONFIG: "1",
    OPENCODE_DISABLE_PROJECT_CONFIG: "1",
    NO_PROXY: "127.0.0.1,localhost",
    no_proxy: "127.0.0.1,localhost",
    PWD: ws,
    TERM: "xterm-256color",
    COLORTERM: "truecolor",
    LANG: "en_US.UTF-8",
  }
  const term = TuiTest.ephemeral("lce", { backend: process.env.E2E_BACKEND ?? "xtermjs", timeouts: { text: 20_000, idle: 10_000 } })
  await term.run(UI_BIN, model ? ["-m", "localcode/e2e-model"] : [], { cols: COLS, rows: ROWS, cwd: ws, env })

  const screen = async () => clean(await term.text())
  const api = {
    term,
    ws,
    screen,
    type: (t) => term.type(t),
    key: (...k) => term.press(...k),
    async waitFor(what, timeout = 20_000) {
      const end = Date.now() + timeout
      let last = ""
      while (Date.now() < end) {
        last = await screen()
        if (typeof what === "string" ? last.includes(what) : what.test(last)) return last
        await new Promise((r) => setTimeout(r, 100))
      }
      throw new Error(`timed out waiting for ${what}\n--- screen ---\n${last}`)
    },
    async gone(what, timeout = 20_000) {
      const end = Date.now() + timeout
      while (Date.now() < end) {
        const s = await screen()
        if (!(typeof what === "string" ? s.includes(what) : what.test(s))) return
        await new Promise((r) => setTimeout(r, 100))
      }
      throw new Error(`still showing ${what}\n${await screen()}`)
    },
    /** Wait until the screen is unchanged for `quiet` ms. */
    async settle(quiet = 700) {
      let prev = await screen()
      let since = Date.now()
      const end = Date.now() + 20_000
      while (Date.now() < end) {
        await new Promise((r) => setTimeout(r, 100))
        const cur = await screen()
        if (cur !== prev) {
          prev = cur
          since = Date.now()
        } else if (Date.now() - since >= quiet) return cur
      }
      throw new Error(`screen never settled\n${prev}`)
    },
    async close() {
      await term.closeQuiet()
      if (mock.exitCode === null) mock.kill("SIGTERM")
      rmSync(root, { recursive: true, force: true })
      rmSync(ws, { recursive: true, force: true })
    },
  }
  return api
}
