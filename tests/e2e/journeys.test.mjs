// User journeys through the real localcode-ui in a pseudo-terminal. Each one
// asserts readable text first, then compares the whole settled screen with a
// committed text snapshot (tests/e2e/__snapshots__).
import assert from "node:assert/strict"
import { mkdirSync } from "node:fs"
import { test } from "node:test"
import { expectSnapshot, open } from "./harness.mjs"

// The picker lists models once the catalog loads. waitFor returns as soon as the
// row shows; the ceiling only matters on a slow shared runner (mock downloads
// take 12 s).
const PICKER_WAIT = 90_000

async function journey(name, opts, body) {
  const s = await open(name, opts)
  try {
    await body(s)
  } catch (e) {
    e.message += `\n--- final screen ---\n${await s.screen().catch(() => "")}`
    throw e
  } finally {
    await s.close()
  }
}
const snap = async (s, name) => {
  const screen = await s.settle()
  // Optional picture for humans (CI artifact); never part of pass/fail.
  if (process.env.E2E_PNG === "1") {
    mkdirSync(new URL("./png/", import.meta.url).pathname, { recursive: true })
    await s.term.screenshot(new URL(`./png/${name}.png`, import.meta.url).pathname).catch((e) => console.warn("png failed:", e.message))
  }
  expectSnapshot(name, screen)
}

test("welcome screen with a model loaded", () =>
  journey("welcome", {}, async (s) => {
    await s.waitFor("localcode · e2e-model")
    const screen = await s.settle()
    assert.match(screen, /Agentic coding\. Local models\. On your Mac\./)
    assert.match(screen, /Ask anything/)
    assert.match(screen, /\/ commands/)
    await snap(s, "welcome")
  }))

test("welcome screen with no model loaded", () =>
  journey("welcome-nomodel", { model: false }, async (s) => {
    await s.waitFor("/ commands")
    const screen = await s.settle()
    assert.match(screen, /Agentic coding\. Local models\./)
    assert.doesNotMatch(screen, /e2e-model/)
    await snap(s, "welcome-no-model")
  }))

test("command palette opens on /", () =>
  journey("palette", {}, async (s) => {
    await s.waitFor("localcode · e2e-model")
    await s.type("/")
    await s.waitFor("/cd")
    const screen = await s.settle()
    for (const c of ["/clear", "/diff", "/exit"]) assert.ok(screen.includes(c), `palette lists ${c}`)
    await snap(s, "palette")
  }))

test("/models picker and quant list", () =>
  journey("models", {}, async (s) => {
    await s.waitFor("localcode · e2e-model")
    await s.type("/models")
    await s.waitFor("/models")
    await s.key("Enter")
    await s.waitFor("Select a model")
    await s.waitFor("Gamma 2 30B", PICKER_WAIT)
    let screen = await s.settle()
    assert.match(screen, /Alpha 4 12B ★/)
    // The maker column fits the longest maker: "Thinking Machines" must not run into the state.
    assert.match(screen, /Beta 3 8B\s+Thinking Machines {2,}Download/)
    await snap(s, "models-picker")

    await s.key("Enter")
    await s.waitFor("Available to download")
    screen = await s.settle()
    assert.match(screen, /Q4_K_M ★\s+6\.9 GB/)
    assert.match(screen, /Q8_0 max\s+12\.7 GB/)
    assert.match(screen, /Q3_K_M\s+5\.1 GB\s+✓ fits\s+~21\.4 GB/)
    assert.match(screen, /Q4_K_M ★\s+6\.9 GB\s+✓ fits\s+~19\.6 GB/)
    // The largest that fits is selectable and marked; it is under the comfort floor, so it says tight.
    assert.match(screen, /Q8_0 max\s+12\.7 GB\s+✓ fits\s+~6\.8 GB tight/)
    await snap(s, "models-quants")
  }))

test("a prompt is sent and the reply streams in", () =>
  journey("reply", {}, async (s) => {
    await s.waitFor("localcode · e2e-model")
    await s.type("hello there")
    await s.key("Enter")
    await s.waitFor("This reply is fixed so screenshots stay stable.")
    await s.waitFor(/Completed in/)
    const screen = await s.settle()
    assert.match(screen, /Hello from the mock model\./)
    await snap(s, "reply")
  }))

test("a tool call asks permission and runs after approval", () =>
  journey("permission", {}, async (s) => {
    await s.waitFor("localcode · e2e-model")
    await s.type("please read outside the project")
    await s.key("Enter")
    await s.waitFor("Permission required")
    await s.waitFor("Allow once")
    const asking = await s.settle()
    assert.match(asking, /Access external directory \/tmp\/lce-outside/)
    assert.doesNotMatch(asking, /outside notes/)
    await snap(s, "permission-prompt")

    await s.key("Enter")
    await s.waitFor("The command finished.")
    await s.waitFor(/Completed in/)
    const done = await s.settle()
    assert.doesNotMatch(done, /Permission required/)
    assert.match(done, /Explored: 1 read/)
    await snap(s, "permission-approved")
  }))

test("/status dialog", () =>
  journey("status", {}, async (s) => {
    await s.waitFor("localcode · e2e-model")
    await s.type("/status")
    await s.waitFor("/status")
    await s.key("Enter")
    await s.waitFor("32,768 token context")
    const screen = await s.settle()
    assert.match(screen, /Model e2e-model/)
    assert.match(screen, /No MCP servers/)
    await snap(s, "status")
  }))

test("/mcps is a plain list with a hint for adding and importing", () =>
  journey("mcps", {}, async (s) => {
    await s.waitFor("localcode · e2e-model")
    await s.type("/mcps")
    await s.waitFor("/mcps")
    await s.key("Enter")
    await s.waitFor("MCP servers")
    const screen = (await s.settle()).replace(/\s+/g, " ")
    assert.match(screen, /Add: localcode mcp add <name> -- <command>/)
    assert.match(screen, /localcode mcp import/)
    assert.doesNotMatch(screen, /ctrl\+a|ctrl\+d/i)
  }))

test("sidebar shows the session context", () =>
  journey("sidebar", {}, async (s) => {
    await s.waitFor("localcode · e2e-model")
    await s.type("hello there")
    await s.key("Enter")
    await s.waitFor(/Completed in/)
    await s.type("/sidebar")
    await s.waitFor("/show-sidebar")
    await s.key("Enter")
    await s.waitFor("Context")
    const screen = await s.settle()
    assert.match(screen, /108 tokens/)
    await snap(s, "sidebar")
  }))

test("the plan panel follows every plan update within a turn", () =>
  journey("plan", {}, async (s) => {
    await s.waitFor("localcode · e2e-model")
    // The sidebar command exists inside a session, so start one first.
    await s.type("hello there")
    await s.key("Enter")
    await s.waitFor(/Completed in/)
    await s.type("/sidebar")
    await s.waitFor("/show-sidebar")
    await s.key("Enter")
    await s.waitFor("Context")
    await s.type("make a plan")
    await s.key("Enter")
    // The mock waits 2 s after each update, so each state is on screen mid-turn.
    await s.waitFor(/\[•\] Alpha step/)
    await s.waitFor(/\[✓\] Alpha step[\s\S]*\[•\] Beta step/)
    await s.waitFor(/\[✓\] Beta step[\s\S]*\[•\] Gamma step/)
    await s.waitFor(/Completed in/)
    assert.match(await s.settle(), /\[•\] Gamma step/)
  }))

test("a model downloads in the background while the chat keeps working", () =>
  journey("background-download", {}, async (s) => {
    await s.waitFor("localcode · e2e-model")
    await s.type("/models")
    await s.waitFor("/models")
    await s.key("Enter")
    await s.waitFor("Gamma 2 30B", PICKER_WAIT)
    await s.key("Enter")
    await s.waitFor("Available to download")
    await s.settle()
    // Q3_K_M is not downloaded: picking it starts a download.
    await s.type("Q3")
    await s.waitFor(/Q3_K_M\s+5\.1 GB/)
    await s.key("Enter")
    await s.waitFor(/Downloading in the background/)
    // The prompt is not locked while the download runs.
    await s.type("hello there")
    await s.key("Enter")
    await s.waitFor("Hello from the mock model.")
    assert.match(await s.screen(), /e2e-model · downloading alpha-Q3_K_M \d+% · about 3 min left/)
    await s.waitFor(/Download finished/, 90_000)
    assert.match(await s.screen(), /alpha-Q3_K_M is ready\. Open \/models to switch to it\./)
  }))

test("the first download says what is happening and when chat can start", () =>
  journey("first-download", { model: false }, async (s) => {
    await s.waitFor("No model loaded")
    await s.type("/models")
    await s.waitFor("/models")
    await s.key("Enter")
    await s.waitFor("Gamma 2 30B", PICKER_WAIT)
    await s.key("Enter")
    await s.waitFor("Available to download")
    await s.settle()
    await s.type("Q3")
    await s.waitFor(/Q3_K_M\s+5\.1 GB/)
    await s.key("Enter")
    await s.waitFor(/Downloading alpha-Q3_K_M \(5\.1 GB\)\. You can chat once it's loaded\./)
    await s.waitFor(/localcode · Downloading \d+% · about 3 min left/)
    // Sending now explains the wait instead of throwing the picker up again.
    await s.type("hello")
    await s.key("Enter")
    await s.waitFor(/alpha-Q3_K_M is still downloading\s+\(\d+%\s+·\s+about 3 min\s+left\)\.\s+You can chat\s+once it's\s+loaded\./)
    assert.doesNotMatch(await s.screen(), /Select a model/)
  }))

test("commands LocalCode dropped are not offered in the / menu", () =>
  journey("dropped-commands", {}, async (s) => {
    await s.waitFor("localcode · e2e-model")
    // A session, so session commands (/copy, /redo, copy transcript) are registered.
    await s.type("hello there")
    await s.key("Enter")
    await s.waitFor(/Completed in/)
    const offered = async (query) => {
      await s.type(query)
      await new Promise((r) => setTimeout(r, 600))
      const screen = await s.screen()
      for (let i = 0; i < query.length; i++) await s.key("Backspace")
      await new Promise((r) => setTimeout(r, 300))
      return screen
    }
    for (const gone of ["/lsp", "/worktrees", "/btw", "/terminal", "/variants", "/effort", "/reload", "/redo",
                        "/copy-session", "/reopen", "/review", "/close-tab", "/variant", "/sessions", "/continue", "/mode$", "/debug", "/toggle", "/select-terminal",
                        "/show-terminal", "/plugins", "/stash", "/queue", "/stats", "/verbosity",
                        "/quit", "/q$", "/say", "/copy", "/open", "/project$", "/projects"]) {
      // "/mode$" means exactly /mode (not /models); other entries match as prefixes.
      const exact = gone.endsWith("$")
      const name = exact ? gone.slice(0, -1) : gone
      const screen = await offered(name)
      // A menu row is "/name", spaces, then its description (the input line has no description).
      assert.doesNotMatch(screen, new RegExp(`^[ \\t]*${name}${exact ? "" : "[a-z-]*"}[ \\t]{2,}\\S`, "m"), `${name} is still offered`)
    }
    assert.match(await offered("/exp"), /\/export\s+Export session transcript/)
    assert.match(await offered("/mod"), /\/models/)
  }))

// The menu is an allowlist (UI fork: packages/tui/src/util/localcode-menu.ts). Scroll the
// whole real menu, on the home screen and in a session, and fail on anything else, so an
// upstream UI update cannot add commands without someone choosing them.
const MENU = new Set(["bench", "cd", "clear", "compact", "decide", "diff", "exit", "export", "help",
  "mcps", "models", "new", "permissions", "project-context", "rename", "resume",
  "speak", "server", "settings", "show-sidebar", "skills", "status", "themes", "thinking", "undo", "vision"])

async function menuNames(s) {
  const seen = new Set()
  await s.type("/")
  await new Promise((r) => setTimeout(r, 800))
  let stale = 0
  for (let i = 0; i < 120 && stale < 12; i++) {
    const before = seen.size
    for (const m of (await s.screen()).matchAll(/^\s*\/([a-z0-9-]+)[ \t]{2,}\S/gm)) seen.add(m[1])
    stale = seen.size === before ? stale + 1 : 0
    await s.key("Down")
    await new Promise((r) => setTimeout(r, 120))
  }
  await s.key("Escape")
  return seen
}

test("the / menu offers only LocalCode's allowlist", () =>
  journey("menu-allowlist", {}, async (s) => {
    await s.waitFor("localcode · e2e-model")
    const home = await menuNames(s)
    await s.type("hello there")
    await s.key("Enter")
    await s.waitFor(/Completed in/)
    const session = await menuNames(s)
    const extra = [...home, ...session].filter((name) => !MENU.has(name))
    assert.deepEqual(extra, [], `not on the allowlist: ${extra.join(", ")}`)
    for (const name of ["models", "resume", "undo", "exit"]) assert.ok(session.has(name), `/${name} is offered`)
  }))

// #259: an image bound for a model that cannot see it is never sent silently.
const PNG = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFklEQVR4nGM4YWBAEmIY1TCqYfhqAAD1mygQumZKZwAAAABJRU5ErkJggg==", "base64")
const imageFiles = { "README.md": "# demo\n", "shot.png": PNG }
async function attachImage(s) {
  await s.waitFor("localcode · e2e-model")
  // A terminal paste of an image path becomes an attachment.
  await s.term.write(`\x1b[200~${s.ws}/shot.png\x1b[201~`)
  await s.waitFor(/\[Image/)
  await s.type("what is this")
  await s.key("Enter")
}
const chatPosts = async (s) => (await s.posts()).filter((p) => p.endsWith("/chat/completions"))

test("an image without vision support is not sent; the model can download it", () =>
  journey("novision-dl", { files: imageFiles, mockArgs: ["--vision=off-available"] }, async (s) => {
    await attachImage(s)
    await s.waitFor("can't see images yet")
    const screen = await s.settle()
    assert.match(screen, /Download vision support \(0\.9 GB\)/)
    assert.match(screen, /reads images with a separate 0\.9 GB/)
    assert.match(screen, /Same as running \/vision/)
    assert.match(screen, /Switch model/)
    assert.doesNotMatch(screen, /Send without image/)
    assert.deepEqual(await chatPosts(s), [], "no chat request while blocked")
    await s.key("Enter")
    await s.waitFor("Enable images for this model?")
    await s.key("Enter")
    await s.waitFor("Images enabled")
    assert.ok((await s.posts()).includes("/vision/install"), "vision install requested")
    assert.deepEqual(await chatPosts(s), [], "still nothing sent")
  }))

test("an image without vision support and no projector offers switch or an explicit send", () =>
  journey("novision-none", { files: imageFiles, mockArgs: ["--vision=off-unavailable"] }, async (s) => {
    await attachImage(s)
    await s.waitFor("can't see images yet")
    const screen = await s.settle()
    assert.match(screen, /can't read images/)
    assert.match(screen, /Switch model/)
    assert.match(screen, /Send without image/)
    assert.doesNotMatch(screen, /Download vision support/)
    assert.deepEqual(await chatPosts(s), [], "no chat request while blocked")
    await s.key("Down")
    await s.key("Enter")
    await s.waitFor("This reply is fixed so screenshots stay stable.")
    assert.equal((await chatPosts(s)).length >= 1, true)
  }))
