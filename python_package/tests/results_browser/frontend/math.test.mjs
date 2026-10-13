import assert from "node:assert/strict";
import { test } from "node:test";

import { hasMath, mathSegments } from "../../../src/heams/results_browser/static/js/math.js";

test("LaTeX delimiters split text into inline and display math", () => {
  assert.deepEqual(mathSegments("Let \\(t\\) be\n\\[\nx^2\n\\]\nso $$y$$ and $z$."), [
    { text: "Let " },
    { tex: "t", display: false },
    { text: " be\n" },
    { tex: "\nx^2\n", display: true },
    { text: "\nso " },
    { tex: "y", display: true },
    { text: " and " },
    { tex: "z", display: false },
    { text: "." },
  ]);
});

test("dollar amounts and escaped dollars stay text", () => {
  for (const text of ["It costs $5 and $10.", "Paid $ 3 then $4", "a \\$x\\$ b", "$x$5", "$a\n\nb$"]) {
    assert.ok(!hasMath(text), text);
  }
  assert.ok(hasMath("Find $9$ kilometers"));
  assert.ok(hasMath("speed $s+\\frac{1}{2}$ km/h"));
});

test("unclosed or empty delimiters are left alone", () => {
  assert.deepEqual(mathSegments("open \\( only"), [{ text: "open \\( only" }]);
  assert.deepEqual(mathSegments("empty \\(\\) here"), [{ text: "empty \\(\\) here" }]);
  assert.ok(!hasMath("no math at all"));
});
