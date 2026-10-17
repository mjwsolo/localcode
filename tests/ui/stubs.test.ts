import { expect, test } from "bun:test";
import { isStubLine } from "../../src/localcode/ui/plugin/localcode";

test("ordinary code that mentions todos or placeholders is not a stub", () => {
  // Lines from a real todo-list app that were flagged three times in one session.
  for (const line of [
    '"""A small CLI todo list manager."""',
    "todo = {",
    "todos.append(todo)",
    "print(f'Added #{todo[\"id\"]}: {todo[\"task\"]}')",
    '<input placeholder="What needs doing?">',
    "def stub_out(mock):",
    "const placeholderText = 'Search'",
  ]) expect(isStubLine(line), line).toBe(false);
});

test("unfinished-work markers are still caught", () => {
  for (const line of [
    "# TODO: handle errors",
    "// FIXME later",
    "  raise NotImplementedError",
    "throw new Error('not implemented')",
    "// placeholder until the API exists",
    "# stub: returns fake data",
    "<!-- coming soon -->",
  ]) expect(isStubLine(line), line).toBe(true);
});
