/**
 * Shorten the wire history, preserving the complete saved conversation.
 *
 * Works on the runtime's model-request messages (opencode v2 `session.hook("context")`):
 * `{ role, content: [...] }` with `text`, `reasoning`, `tool-call` and `tool-result` parts,
 * tool results arriving in `role: "tool"` messages. Nothing here mutates its input.
 */
const NOTE = "[Earlier tool data abbreviated; reread the relevant file if needed.]";
const KEEP_TOOLS = new Set(["skill", "question"]);
const WORKSPACE_BLOCK = /^WORKSPACE (SNAPSHOT AT SESSION START|CHANGED SINCE YOUR LAST TURN)/;

function brief(value: any, limit: number): any {
  if (typeof value === "string") {
    if (value.length <= limit) return value;
    return value.slice(0, limit - 150) + "\n" + NOTE + "\n" + value.slice(-80);
  }
  if (Array.isArray(value)) return value.map((item) => brief(item, limit));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key,
      ["filePath", "file_path", "path", "workdir"].includes(key) ? item : brief(item, limit),
    ]));
  }
  return value;
}

/** Text parts of a tool result keep their text (bounded); media is dropped from older turns. */
function briefContent(items: any[], limit: number): any[] {
  return items.flatMap((item) => {
    if (item?.type === "text" && typeof item.text === "string") return [{ ...item, text: brief(item.text, limit) }];
    if (item?.type === "media" || item?.type === "file" || item?.type === "image") return [];
    return [item];
  });
}

function briefResult(part: any): any {
  const result = part.result;
  if (!result || typeof result !== "object") return part;
  if (part.resultType === "error" || "error" in result) {
    return { ...part, result: {
      ...result,
      ...(typeof result.error === "string" ? { error: brief(result.error, 400) } : {}),
      ...(result.error && typeof result.error === "object" && typeof result.error.message === "string"
        ? { error: { ...result.error, message: brief(result.error.message, 400) } } : {}),
      ...(Array.isArray(result.content) ? { content: briefContent(result.content, 800) } : {}),
    } };
  }
  if (result.type === "text" && typeof result.value === "string") return { ...part, result: { ...result, value: brief(result.value, 800) } };
  if (result.type === "content" && Array.isArray(result.value)) return { ...part, result: { ...result, value: briefContent(result.value, 800) } };
  if (result.type === "json") return { ...part, result: { ...result, value: brief(result.value, 800) } };
  return part;
}

/** Copy a message with new content, keeping its class (the runtime re-validates messages). */
function withContent(message: any, content: any[]): any {
  return Object.assign(Object.create(Object.getPrototypeOf(message)), message, { content });
}

export function leanMessages(messages: any[], workspaceContext = false): any[] {
  const lastUser = messages.findLastIndex((message) => message?.role === "user");
  return messages.map((message, index) => {
    if (!Array.isArray(message?.content)) return message;
    if (message.role === "user") {
      if (workspaceContext) return message;
      // Automatic workspace context is only ever added by LocalCode, as its own text part.
      const content = message.content.filter((part: any) =>
        !(part?.type === "text" && part.metadata?.localcode === true && WORKSPACE_BLOCK.test(part.text)));
      return content.length === message.content.length ? message : withContent(message, content);
    }
    // Keep all current-turn data, including tools needed to finish the task.
    if (index >= lastUser) return message;
    if (message.role === "assistant") {
      return withContent(message, message.content.flatMap((part: any) => {
        if (part?.type === "reasoning") return [];
        // Explicitly loaded workflows and user decisions remain authoritative.
        if (part?.type !== "tool-call" || KEEP_TOOLS.has(part.name)) return [part];
        return [{ ...part, input: brief(part.input, 600) }];
      }));
    }
    if (message.role === "tool") {
      return withContent(message, message.content.map((part: any) =>
        part?.type !== "tool-result" || KEEP_TOOLS.has(part.name) ? part : briefResult(part)));
    }
    return message;
  });
}

/**
 * Observation masking: mask first, summarize late.
 *
 * Every edit to an old message makes the server re-read everything after it, and the hybrid
 * (Qwen 3.6) and sliding-window (Gemma 4) models re-read the whole prompt when no cache
 * checkpoint precedes the edit. So history is left alone until the prompt nears the context
 * budget; then, in one large batch, every completed tool result outside a protected window of
 * recent output is replaced by a one-line stub naming the tool, its arguments and a file holding
 * the full output, so the model can read it again. Masked results stay masked, and a pass never
 * reaches back past an already-masked result, so the cached prefix before the batch is kept.
 * Only when masking cannot bring the prompt down does the runtime's compaction summarize.
 * All thresholds are shares of the measured budget, never fixed token counts.
 */
const MASK_AT = 0.7;          // start a batch when the prompt reaches this share of the budget
const MASK_PROTECT = 0.25;    // newest tool output kept verbatim, as a share of the budget
const MASK_MINIMUM = 0.125;   // a batch must reclaim at least this share, or nothing changes
const MASK_PROTECTED_TOOLS = new Set(["skill"]);
/** Where masked outputs are saved, relative to the project directory. */
export const MASK_DIR = ".localcode-agent/tool-out";

export type MaskLimits = { trigger: number; protect: number; minimum: number };

export function maskLimits(inputBudget: number): MaskLimits | undefined {
  if (!Number.isFinite(inputBudget) || inputBudget <= 0) return undefined;
  return {
    trigger: Math.floor(inputBudget * MASK_AT),
    protect: Math.floor(inputBudget * MASK_PROTECT),
    minimum: Math.floor(inputBudget * MASK_MINIMUM),
  };
}

export function resultText(result: any): string {
  if (!result || typeof result !== "object") return "";
  if (typeof result.value === "string") return result.value;
  if (Array.isArray(result.value)) return result.value.map((item: any) => (item?.type === "text" ? item.text : "")).join("");
  return result.value === undefined ? "" : JSON.stringify(result.value);
}

export const estimate = (text: string) => Math.ceil(text.length / 4);

/** Rough size of a request as the server will see it (calibrated against measured counts by the caller). */
export function estimateRequest(messages: any[], system: any[] = [], tools: Record<string, any> = {}): number {
  let chars = 0;
  for (const part of system) chars += String(typeof part === "string" ? part : part?.text ?? "").length;
  for (const [name, tool] of Object.entries(tools)) chars += name.length + String(tool?.description ?? "").length + JSON.stringify(tool?.input ?? tool?.inputSchema ?? {}).length;
  for (const message of messages) {
    if (typeof message?.content === "string") { chars += message.content.length; continue; }
    for (const part of message?.content ?? []) {
      if (part?.type === "text" || part?.type === "reasoning") chars += String(part.text ?? "").length;
      else if (part?.type === "tool-call") chars += String(part.name ?? "").length + (JSON.stringify(part.input) ?? "").length;
      else if (part?.type === "tool-result") chars += resultText(part.result).length + (typeof part.result?.error === "string" ? part.result.error.length : 0);
    }
  }
  return Math.ceil(chars / 4);
}

const isFailed = (item: any) => item?.resultType === "error" || "error" in (item?.result ?? {});

/**
 * Tool results to mask now (ids), given the ones already masked and the prompt's current size.
 * Empty until the prompt reaches the trigger, or when the batch would reclaim too little.
 */
export function maskCandidates(messages: any[], limits: MaskLimits | undefined, masked: ReadonlySet<string>, promptTokens: number): string[] {
  if (!limits || promptTokens < limits.trigger) return [];
  let total = 0;
  let reclaim = 0;
  const found: string[] = [];
  loop: for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    if (!Array.isArray(message?.content)) continue;
    for (let part = message.content.length - 1; part >= 0; part--) {
      const item = message.content[part];
      if (item?.type !== "tool-result" || isFailed(item) || MASK_PROTECTED_TOOLS.has(item.name)) continue;
      if (masked.has(item.id)) break loop;
      const size = estimate(resultText(item.result));
      total += size;
      if (total <= limits.protect) continue;
      reclaim += size;
      found.push(item.id);
    }
  }
  return reclaim >= limits.minimum && reclaim > 0 ? found : [];
}

/** The saved-output file for a tool call id, relative to the project directory. */
export const maskPath = (id: string) => `${MASK_DIR}/${String(id).replace(/[^A-Za-z0-9_-]/g, "_")}.txt`;

const ARG_KEYS = ["filePath", "file_path", "path", "pattern", "command", "url", "query", "include", "glob"];

/** A short, deterministic summary of a tool call's arguments. */
export function argsSummary(input: any): string {
  if (!input || typeof input !== "object") return "";
  const keys = [...ARG_KEYS.filter((key) => key in input), ...Object.keys(input).filter((key) => !ARG_KEYS.includes(key))];
  const parts: string[] = [];
  for (const key of keys) {
    const value = input[key];
    if (value === undefined || value === null || typeof value === "object") continue;
    const text = String(value).replace(/\s+/g, " ").trim();
    parts.push(`${key}=${text.length > 80 ? text.slice(0, 77) + "..." : text}`);
    if (parts.join(", ").length > 160) break;
  }
  const out = parts.join(", ");
  return out.length > 200 ? out.slice(0, 197) + "..." : out;
}

export function maskStub(name: string, input: any, id: string): string {
  return `[Old tool output masked to save context: ${name}(${argsSummary(input)}). Full output saved in ${maskPath(id)}; read that file if you need it again.]`;
}

/** Replace masked tool results with their stubs; nothing else changes. */
export function applyMasked(messages: any[], masked: ReadonlySet<string>): any[] {
  if (!masked.size) return messages;
  const inputs = new Map<string, any>();
  for (const message of messages) {
    if (message?.role !== "assistant" || !Array.isArray(message.content)) continue;
    for (const part of message.content) if (part?.type === "tool-call") inputs.set(part.id, part.input);
  }
  return messages.map((message) => {
    if (!Array.isArray(message?.content) || !message.content.some((item: any) => item?.type === "tool-result" && masked.has(item.id))) return message;
    return withContent(message, message.content.map((item: any) =>
      item?.type === "tool-result" && masked.has(item.id)
        ? { ...item, result: { type: "text", value: maskStub(String(item.name ?? "tool"), inputs.get(item.id), item.id) } }
        : item));
  });
}

/** Text of the tool results with these ids, for saving before they are masked. */
export function maskedOutputs(messages: any[], ids: readonly string[]): Map<string, string> {
  const wanted = new Set(ids);
  const out = new Map<string, string>();
  for (const message of messages) {
    if (!Array.isArray(message?.content)) continue;
    for (const item of message.content) if (item?.type === "tool-result" && wanted.has(item.id)) out.set(item.id, resultText(item.result));
  }
  return out;
}

/**
 * After a compaction the request opens with the runtime's checkpoint message. Open todos at the
 * time of the compaction ride at its tail (snapshot frozen per checkpoint, so the prefix is stable).
 */
export const CHECKPOINT_START = "<conversation-checkpoint>";

export function checkpointIndex(messages: any[]): number {
  return messages.findIndex((message) => {
    if (message?.role !== "user") return false;
    const first = typeof message.content === "string" ? message.content : message.content?.find?.((part: any) => part?.type === "text")?.text;
    return typeof first === "string" && first.startsWith(CHECKPOINT_START);
  });
}

export function withCheckpointTodos(messages: any[], index: number, text: string): any[] {
  if (index < 0 || !text) return messages;
  const message = messages[index];
  const content = typeof message.content === "string" ? [{ type: "text", text: message.content }] : [...message.content];
  return messages.map((item, i) => i === index ? withContent(item, [...content, { type: "text", text, metadata: { localcode: true } }]) : item);
}
