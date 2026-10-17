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
