import assert from "node:assert/strict";
import { test } from "node:test";

import { escapeHtml, html, Markup } from "../../../src/heams/results_browser/static/js/dom.js";

test("escapeHtml escapes markup and quotes", () => {
  assert.equal(escapeHtml(`<a href="x" onclick='y'>&</a>`), "&lt;a href=&quot;x&quot; onclick=&#39;y&#39;&gt;&amp;&lt;/a&gt;");
  assert.equal(escapeHtml(null), "");
  assert.equal(escapeHtml(42), "42");
});

test("html escapes interpolated data", () => {
  const name = "<img src=x onerror=alert(1)>";
  assert.equal(html`<b>${name}</b>`.toString(), "<b>&lt;img src=x onerror=alert(1)&gt;</b>");
});

test("html keeps nested markup and renders arrays and empty values", () => {
  const items = ["a<", "b"].map((item) => html`<li>${item}</li>`);
  const result = html`<ul>${items}</ul>${null}${undefined}${false}`;
  assert.ok(result instanceof Markup);
  assert.equal(result.toString(), "<ul><li>a&lt;</li><li>b</li></ul>");
});

test("html escapes attribute values", () => {
  const title = `x" onmouseover="alert(1)`;
  assert.equal(html`<td title="${title}"></td>`.toString(), `<td title="x&quot; onmouseover=&quot;alert(1)"></td>`);
});
