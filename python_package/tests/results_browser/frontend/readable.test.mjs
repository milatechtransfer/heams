import assert from "node:assert/strict";
import { test } from "node:test";

import {
  expandValue,
  layout,
  looksEscaped,
  parsePythonLiteral,
  readableText,
  unescapeText,
} from "../../../src/heams/results_browser/static/js/readable.js";

test("a repr stored inside a judge reason is decoded", () => {
  // As EvoMemory recorded it: the model answer as a Python repr.
  const reason = "expected AIME answer '204', got 'Let \\\\(t\\\\) be the time.\\n\\\\[\\n\\\\boxed{214}\\n\\\\]'";
  assert.ok(looksEscaped(reason));
  assert.equal(readableText(reason), "expected AIME answer '204', got 'Let \\(t\\) be the time.\n\\[\n\\boxed{214}\n\\]'");
});

test("text that already has line breaks, or LaTeX, is left alone", () => {
  assert.ok(!looksEscaped("a\nb \\n c \\n"));
  assert.ok(!looksEscaped("x \\neq y"));
  assert.equal(readableText("x \\neq y"), "x \\neq y");
});

test("escapes decode left to right", () => {
  assert.equal(unescapeText("\\\\neq\\n\\u00e9\\'"), "\\neq\né'");
});

test("JSON inside string values is parsed", () => {
  assert.deepEqual(expandValue({ output: '{"hits": [{"text": "a"}]}' }), { output: { hits: [{ text: "a" }] } });
  assert.equal(expandValue("[not json"), "[not json");
});

test("structures read as indented key: value lines with real line breaks", () => {
  const value = { memory_calls: [{ arguments: { content: "line 1\nline 2", k: 5 } }], empty: [], none: null };
  assert.equal(
    layout(value),
    ["memory_calls:", "  -", "    arguments:", "      content:", "        line 1", "        line 2", "      k: 5", "empty: []", "none: null"].join("\n"),
  );
  assert.equal(layout([1, { _truncated_items: 3 }]), "- 1\n… 3 more items");
});

test("Python str() of tool output parses, including JSON inside its strings", () => {
  const output = `['{"content": "It\\'s great", "metadata": {"order": 2}}', ("a", 1.5, True, None)]`;
  assert.deepEqual(parsePythonLiteral(output), ['{"content": "It\'s great", "metadata": {"order": 2}}', ["a", 1.5, true, null]]);
  assert.deepEqual(expandValue(output), [{ content: "It's great", metadata: { order: 2 } }, ["a", 1.5, true, null]]);
  assert.deepEqual(parsePythonLiteral("{'k': [], 1: {}}"), { k: [], 1: {} });
  for (const bad of ["[1, 2", "['open]", "[x]", "[1] tail"]) assert.throws(() => parsePythonLiteral(bad), SyntaxError);
  assert.equal(expandValue("[see note 3]"), "[see note 3]");
});

test("terminal color codes are dropped", () => {
  assert.equal(readableText("be \u001b[1mA\u001b[0m."), "be A.");
  assert.equal(readableText("x\\n\\x1b[1mB\\x1b[0m\\n"), "x\nB\n");
});
