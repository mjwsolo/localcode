import { expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Plugin from "../../src/localcode/ui/plugin/localcode";

test("actual length finish continues once; guessed limits and interruption do not", async () => {
  const directory = mkdtempSync(join(tmpdir(), "output-limit-"));
  const prompts: any[] = [];
  const warnings: any[] = [];
  try {
    const hooks = await (Plugin as any)({directory, client:{session:{prompt: async (p:any) => {prompts.push(p)}},tui:{showToast:async (p:any) => {warnings.push(p)}}}});
    await hooks["chat.message"]({sessionID:"s",agent:"build"}, {parts:[{type:"text",text:"Write the complete answer"}]});
    await hooks.event({event:{type:"session.idle",properties:{sessionID:"s"}}});
    expect(prompts).toHaveLength(0);
    for (const id of ["one", "two"]) {
      await hooks.event({event:{type:"message.part.updated",properties:{part:{id,type:"step-finish",reason:"length",sessionID:"s"}}}});
      await hooks.event({event:{type:"session.idle",properties:{sessionID:"s"}}});
    }
    expect(prompts).toHaveLength(1);
    expect(warnings).toHaveLength(1);
    expect(warnings[0].body.message).toContain("may be incomplete");
    expect(prompts[0].body.parts[0].text).toContain("runtime reported");
    await hooks["chat.message"]({sessionID:"s",agent:"build"}, {parts:[{type:"text",text:"next"}]});
    await hooks.event({event:{type:"session.error",properties:{sessionID:"s",error:{name:"MessageAbortedError"}}}});
    await hooks.event({event:{type:"message.part.updated",properties:{part:{id:"three",type:"step-finish",reason:"length",sessionID:"s"}}}});
    await hooks.event({event:{type:"session.idle",properties:{sessionID:"s"}}});
    expect(prompts).toHaveLength(1);
  } finally {rmSync(directory,{recursive:true,force:true})}
});
