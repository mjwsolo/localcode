/**
 * The channel from LocalCode's server plugin to its terminal UI entry. The server
 * side has no screen; notifications (a stopped task, a crashed model server, an
 * output limit) are emitted as RPC events that `tui.ts` shows as toasts.
 */
export const Definition = {
  id: "localcode",
  methods: {},
  events: {
    toast: {
      schema: {
        type: "object",
        properties: {
          title: { type: "string" },
          message: { type: "string" },
          variant: { type: "string", enum: ["info", "success", "warning", "error"] },
          duration: { type: "number" },
        },
        required: ["message", "variant"],
      },
    },
  },
} as const;

export type Toast = { title?: string; message: string; variant: "info" | "success" | "warning" | "error"; duration?: number };

export * as LocalcodeRpc from "./rpc";
