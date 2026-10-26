// User journeys through the real localcode-ui in a pseudo-terminal. Each one
// asserts readable text first, then compares the whole settled screen with a
// committed text snapshot (tests/e2e/__snapshots__).
import assert from "node:assert/strict"
import { mkdirSync } from "node:fs"
import { test } from "node:test"
import { expectSnapshot, open } from "./harness.mjs"

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
    await s.waitFor("/btw")
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
    await s.waitFor("Gamma 2 30B")
    let screen = await s.settle()
    assert.match(screen, /Alpha 4 12B ★/)
    assert.match(screen, /Beta 3 8B\s+Globex\s+Download/)
    await snap(s, "models-picker")

    await s.key("Enter")
    await s.waitFor("Available to download")
    screen = await s.settle()
    assert.match(screen, /Q4_K_M ★\s+6\.9 GB/)
    assert.match(screen, /Q8_0\s+12\.7 GB/)
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
