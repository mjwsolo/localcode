/**
 * localcode branding — header, footer, window title, and a system prompt with
 * no upstream-harness references. All via pi's native setHeader/setFooter APIs;
 * no rendering code of our own beyond returning strings.
 *
 * Footer layout: "localcode" pinned bottom-LEFT, everything else right-aligned.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BRAND = "localcode";

export default function (pi: ExtensionAPI) {
  pi.on("session_start", (_e, ctx) => {
    if (!ctx.hasUI || ctx.mode !== "tui") return;

    ctx.ui.setTitle(BRAND);

    ctx.ui.setHeader((_tui, theme) => ({
      invalidate() {},
      render(_width: number): string[] {
        return [
          ` ${BRAND}`,
          ` a coding agent running a local model on your Mac`,
          "",
        ];
      },
    }));

    ctx.ui.setFooter((tui, theme, footerData) => ({
      invalidate() {},
      render(width: number): string[] {
        const left = BRAND;
        const bits: string[] = [];
        const model = ctx.model?.id;
        if (model) bits.push(model);
        const branch = footerData.getGitBranch();
        if (branch) bits.push(branch);
        for (const [, status] of footerData.getExtensionStatuses()) {
          if (status) bits.push(status);
        }
        const right = bits.join("  ·  ");
        const pad = Math.max(1, width - left.length - right.length - 2);
        return [` ${left}${" ".repeat(pad)}${right} `];
      },
      dispose: footerData.onBranchChange(() => tui.requestRender()),
    }));
  });

  // Strip the upstream harness's self-documentation out of the system prompt:
  // it points the model at that project's docs, which is noise in a user's repo.
  pi.on("before_agent_start", (event) => {
    const o = event.systemPromptOptions;
    if (!o || o.customPrompt) return;
    o.customPrompt =
      `You are ${BRAND}, a coding assistant running a local model on the user's own machine. ` +
      `You help by reading files, running commands, editing code, and writing new files.\n\n` +
      `Guidelines:\n` +
      `- Use bash for file operations like ls, rg, find\n` +
      `- Be concise in your responses\n` +
      `- Show file paths clearly when working with files`;
  });
}
