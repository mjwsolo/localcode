/**
 * The terminal UI entry of LocalCode's plugin directory (opencode v2 loads it for the
 * server plugin in the same directory). It shows the server plugin's notifications as
 * toasts and adds the LocalCode modes (/mode, /decide, /server).
 */
import { LocalcodeRpc, type Toast } from "./rpc";
import { setupModes, type ModesContext } from "./modes";

type TuiContext = ModesContext & {
  readonly client: { rpc(definition: typeof LocalcodeRpc.Definition): { events: { on(name: "toast", handler: (event: { data: Toast }) => void): () => void } } };
};

export function setupToasts(context: TuiContext): () => void {
  try {
    return context.client.rpc(LocalcodeRpc.Definition).events.on("toast", (event) => {
      const toast = event.data;
      if (!toast?.message) return;
      context.ui.toast.show({
        message: toast.message,
        variant: toast.variant,
        ...(toast.title ? { title: toast.title } : {}),
        ...(toast.duration ? { duration: toast.duration } : {}),
      });
    });
  } catch {
    return () => {};
  }
}

export default {
  id: "localcode",
  setup(context: TuiContext) {
    const stopToasts = setupToasts(context);
    const stopModes = setupModes(context);
    return () => {
      stopToasts();
      stopModes();
    };
  },
};
