/** Shorten the wire history, preserving the complete saved conversation. */
const NOTE = "[Earlier tool data abbreviated; reread the relevant file if needed.]";

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

export function leanHistory(messages: any[], workspaceContext = false): any[] {
  const lastUser = messages.findLastIndex((message) => message.info?.role === "user");
  return messages.map((message, index) => ({
    ...message,
    parts: message.parts.flatMap((part: any) => {
      if (!workspaceContext && part.synthetic && part.type === "text" &&
          /^WORKSPACE (SNAPSHOT AT SESSION START|CHANGED SINCE YOUR LAST TURN)/.test(part.text)) return [];
      // Keep all current-turn data, including tools needed to finish the task.
      if (index >= lastUser || message.info?.role !== "assistant") return [part];
      if (part.type === "reasoning") return [];
      // Explicitly loaded workflows and user decisions remain authoritative.
      if (part.type !== "tool" || !part.state || ["skill", "question"].includes(part.tool)) return [part];
      return [{ ...part, state: {
        ...part.state,
        input: brief(part.state.input, 600),
        ...(typeof part.state.output === "string" ? { output: brief(part.state.output, 800) } : {}),
        ...(typeof part.state.error === "string" ? { error: brief(part.state.error, 400) } : {}),
        ...(typeof part.state.metadata?.output === "string" ? {
          metadata: { ...part.state.metadata, output: brief(part.state.metadata.output, 800) },
        } : {}),
        ...(part.state.attachments ? { attachments: [] } : {}),
      } }];
    }),
  }));
}
