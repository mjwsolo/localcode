import { describe, expect, test } from "bun:test";
import LocalcodePluginDefault from "../../src/localcode/ui/plugin/localcode";

const { bashDescription, trimToolDefinition } = LocalcodePluginDefault as any;

// A stand-in for the runtime's bash description: the facts the model needs are
// embedded in the same phrases the real text uses.
const UPSTREAM_BASH = [
  "Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.",
  "Use `/tmp/lc-scratch` for temporary work outside the workspace. This directory has already been created, already exists, and is pre-approved for external directory access.",
  "Before executing the command, please follow these steps:\n1. Directory Verification: ...\n2. Command Execution: ...",
  "  - You can specify an optional timeout in milliseconds. If not specified, commands will time out after 120000ms.",
  "  - If the output exceeds 2000 lines or 51200 bytes, it will be truncated and the full output will be written to a file.",
  "# Git and GitHub\n- Only commit, amend, push, or create PRs when explicitly requested.",
].join("\n\n") + "x".repeat(3000);

describe("tool definitions sent to the model", () => {
  test("bash description keeps the facts and drops the lecture", () => {
    const d = bashDescription(UPSTREAM_BASH);
    expect(d.length).toBeLessThan(800);
    expect(d).toContain("/tmp/lc-scratch");
    expect(d).toContain("120 s");
    expect(d).toContain("2000 lines or 51200 bytes");
    expect(d).toContain("workdir");
    expect(d).toContain("only commit, push or open PRs when asked");
    expect(d).not.toContain("Directory Verification");
  });

  test("the hook rewrites only a long bash description", () => {
    const bash = { description: UPSTREAM_BASH };
    trimToolDefinition("bash", bash);
    expect(bash.description.length).toBeLessThan(800);
    const read = { description: "Reads a file." };
    trimToolDefinition("read", read);
    expect(read.description).toBe("Reads a file.");
    const short = { description: "Runs bash." };
    trimToolDefinition("bash", short);
    expect(short.description).toBe("Runs bash.");
  });
});
