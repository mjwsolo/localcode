/**
 * launch_app — "scaffolds and launches apps, then checks that they respond".
 *
 * localcode's README promises this and pi has nothing like it. Leaner than
 * localcode's launcher.py, but the user-visible contract is the same: detect
 * how to start the project, run it in the background, poll until the port
 * actually answers, and report the URL, PID and log path. It never blocks the
 * agent on a server that never exits.
 */
import { execFileSync, spawn } from "node:child_process";
import { existsSync, openSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

type Plan = { cmd: string; args: string[]; hint: string };

/** How does this project start? Mirrors the cases localcode's launcher handles. */
function detect(root: string): Plan | null {
  const pkgPath = join(root, "package.json");
  if (existsSync(pkgPath)) {
    try {
      const scripts = JSON.parse(readFileSync(pkgPath, "utf8")).scripts ?? {};
      for (const s of ["dev", "start", "serve", "preview"]) {
        if (scripts[s]) return { cmd: "npm", args: ["run", s], hint: `npm run ${s}` };
      }
    } catch {}
  }
  if (existsSync(join(root, "manage.py")))
    return { cmd: "python3", args: ["manage.py", "runserver"], hint: "python3 manage.py runserver" };
  for (const f of ["app.py", "main.py", "server.py"]) {
    if (existsSync(join(root, f))) return { cmd: "python3", args: [f], hint: `python3 ${f}` };
  }
  if (existsSync(join(root, "Cargo.toml"))) return { cmd: "cargo", args: ["run"], hint: "cargo run" };
  if (existsSync(join(root, "go.mod"))) return { cmd: "go", args: ["run", "."], hint: "go run ." };
  return null;
}

const PORTS = [3000, 5173, 8000, 8080, 4200, 8501, 1420, 3001, 4321];

/** Ports THIS process tree is listening on — macOS squats on 5000 (AirPlay),
 *  so guessing from a fixed list produces false positives. Ask lsof instead. */
function listeningPorts(pid: number): number[] {
  try {
    const out = execFileSync("lsof", ["-nP", "-iTCP", "-sTCP:LISTEN", "-a", "-g", String(pid)],
                             { encoding: "utf8", timeout: 4000 });
    return [...new Set([...out.matchAll(/:(\d+)\s+\(LISTEN\)/g)].map((m) => Number(m[1])))];
  } catch { return []; }
}

async function answering(port: number): Promise<boolean> {
  try {
    const r = await fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(1500) });
    return r.status < 600;
  } catch { return false; }
}

const launchApp = defineTool({
  name: "launch_app",
  label: "Launch app",
  description:
    "Start this project's dev server in the background and verify it responds. " +
    "Returns the URL, PID and log path. Use after scaffolding an app.",
  promptSnippet: "launch_app: start the project's dev server and check that it responds",
  parameters: Type.Object({
    directory: Type.Optional(Type.String({ description: "Project root (default: cwd)" })),
    port: Type.Optional(Type.Number({ description: "Expected port, if you know it" })),
  }),
  async execute(_id, params, _signal, _onUpdate, ctx) {
    const root = params.directory ?? ctx.cwd;
    const plan = detect(root);
    if (!plan) {
      return {
        content: [{ type: "text", text:
          `No start command found in ${root}. Looked for a package.json dev/start/serve/preview script, ` +
          `manage.py, app.py/main.py/server.py, Cargo.toml or go.mod.` }],
        details: { ok: false, url: "", pid: 0, log: "" },
      };
    }
    const log = join(tmpdir(), `localcode-app-${Date.now()}.log`);
    const fd = openSync(log, "a");
    const child = spawn(plan.cmd, plan.args, { cwd: root, detached: true, stdio: ["ignore", fd, fd] });
    child.unref();

    const candidates = params.port ? [params.port, ...PORTS] : PORTS;
    const deadline = Date.now() + 60_000;
    while (Date.now() < deadline) {
      if (child.exitCode !== null) break;
      const owned = child.pid ? listeningPorts(child.pid) : [];
      for (const p of [...owned, ...candidates]) {
        if (await answering(p)) {
          return {
            content: [{ type: "text", text:
              `Started with \`${plan.hint}\` and it responded.\nURL: http://localhost:${p}\nPID: ${child.pid}\nLog: ${log}` }],
            details: { ok: true, url: `http://localhost:${p}`, pid: child.pid ?? 0, log },
          };
        }
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
    let tail = "";
    try { tail = readFileSync(log, "utf8").split("\n").slice(-25).join("\n"); } catch {}
    return {
      content: [{ type: "text", text:
        `Ran \`${plan.hint}\` but nothing answered on ${candidates.slice(0, 6).join(", ")} within 60s.\n` +
        `PID: ${child.pid}\nLog: ${log}\n\nLast log lines:\n${tail}` }],
      details: { ok: false, url: "", pid: child.pid ?? 0, log },
    };
  },
});

export default function (pi: ExtensionAPI) {
  pi.registerTool(launchApp);
}
