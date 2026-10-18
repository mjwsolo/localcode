/**
 * Lazy image projector. The model server starts text-only, so a session that
 * never sends an image never spends the projector's memory (0.2-3.9 GB per
 * model). The first request that carries an image (attached by the user or
 * returned by the read tool) asks the supervisor to restart the server with
 * the projector, after a one-time notice, and waits until it is ready.
 *
 * Nothing is downloaded here: a projector that is not on disk stays the
 * ask-first `/vision` flow.
 */

const isImage = (type: unknown) => typeof type === "string" && type.toLowerCase().startsWith("image/");

/** True when a model-request message list (opencode v2 `session.hook("context")`) carries an image. */
export function hasImage(messages: any[]): boolean {
  const scan = (items: unknown): boolean => Array.isArray(items) && items.some((item: any) => {
    if (!item || typeof item !== "object") return false;
    if (item.type === "image") return true;
    if ((item.type === "media" || item.type === "file") && (isImage(item.mediaType) || isImage(item.mime))) return true;
    if (item.type === "tool-result" && item.result && typeof item.result === "object") return scan(item.result.value) || scan(item.result.content);
    return false;
  });
  return messages.some((message) => scan(message?.content));
}

type Status = { state?: string; current?: string | null; vision?: boolean; vision_loaded?: boolean; vision_available?: boolean; vision_size_gb?: number; detail?: string };
type Toast = (input: { body: { title?: string; message: string; variant: string; duration?: number } }) => Promise<unknown> | unknown;

export type VisionDeps = {
  control: string | undefined;
  headers: () => Record<string, string>;
  toast: Toast;
  fetch?: typeof fetch;
  sleep?: (ms: number) => Promise<void>;
  timeoutMs?: number;
};

export function createVisionLoader(deps: VisionDeps) {
  const http = deps.fetch ?? fetch;
  const sleep = deps.sleep ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  const noticed = new Set<string>();
  let pending: Promise<void> | undefined;

  async function status(): Promise<Status | undefined> {
    try {
      const response = await http(`${deps.control}/status`, { headers: deps.headers(), signal: AbortSignal.timeout(2_000) });
      return response.ok ? (await response.json()) as Status : undefined;
    } catch {
      return undefined;
    }
  }

  async function load(): Promise<void> {
    const before = await status();
    if (!before?.current || before.vision_loaded) return;
    if (!before.vision) {
      if (before.vision_available && !noticed.has(`missing:${before.current}`)) {
        noticed.add(`missing:${before.current}`);
        await deps.toast({ body: { title: "Images", message: `${before.current} needs its ${before.vision_size_gb ?? "?"} GB image projector to read images. Run /vision to download it.`, variant: "warning", duration: 10_000 } });
      }
      return;
    }
    let started: { ok?: boolean; error?: string; loaded?: boolean } = {};
    try {
      const response = await http(`${deps.control}/vision/load`, { method: "POST", headers: deps.headers(), body: "{}" });
      started = await response.json() as typeof started;
    } catch (error: any) {
      started = { error: error?.message ?? String(error) };
    }
    if (started.error) {
      await deps.toast({ body: { title: "Images", message: started.error, variant: "error", duration: 10_000 } });
      return;
    }
    if (started.loaded) return;
    if (!noticed.has(before.current)) {
      noticed.add(before.current);
      await deps.toast({ body: { title: "Images", message: `Loading the ${before.vision_size_gb ?? "?"} GB image projector for ${before.current}. The model restarts once and reads the conversation again.`, variant: "info", duration: 10_000 } });
    }
    const deadline = Date.now() + (deps.timeoutMs ?? 300_000);
    while (Date.now() < deadline) {
      await sleep(500);
      const now = await status();
      if (!now) continue;
      if (now.state === "ready" && now.vision_loaded) return;
      if (now.state === "error") {
        await deps.toast({ body: { title: "Images", message: now.detail || "The image projector did not load", variant: "error", duration: 10_000 } });
        return;
      }
    }
  }

  /** Make sure the projector is loaded before a request with an image goes out. */
  return function ensureVision(messages: any[]): Promise<void> {
    if (!deps.control || !hasImage(messages)) return Promise.resolve();
    // One load at a time: concurrent requests (title, compaction) share it.
    pending ??= load().catch(() => {}).finally(() => { pending = undefined; });
    return pending;
  };
}
