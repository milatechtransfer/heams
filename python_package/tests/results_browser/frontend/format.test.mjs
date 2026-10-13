import assert from "node:assert/strict";
import { test } from "node:test";

import {
  formatBytes,
  formatDuration,
  formatPercent,
  formatValue,
  heatLevel,
  lowerIsBetter,
} from "../../../src/heams/results_browser/static/js/format.js";

test("values are formatted in their unit", () => {
  assert.equal(formatValue(72.4705, "%"), "72.5%");
  assert.equal(formatValue(1.5, "s"), "1.50 s");
  assert.equal(formatValue(1234567.4, "tokens"), "1,234,567");
  assert.equal(formatValue(null, "%"), "N/A");
  assert.equal(formatPercent(0.5), "50.0%");
  assert.equal(formatPercent(null), "–");
});

test("lower is better for costs and error rates only", () => {
  assert.ok(lowerIsBetter("s", "Duration"));
  assert.ok(lowerIsBetter("%", "Memory operation error rate"));
  assert.ok(!lowerIsBetter("%", "Overall"));
});

test("durations and sizes", () => {
  assert.equal(formatDuration(5.25), "5.3 ms");
  assert.equal(formatDuration(250), "250 ms");
  assert.equal(formatDuration(13930), "13.93 s");
  assert.equal(formatDuration(125000), "2m 5s");
  assert.equal(formatDuration(undefined), "–");
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(3.6 * 1024 ** 3), "3.6 GB");
});

test("heat level applies to percentages only", () => {
  assert.equal(heatLevel(50, "%"), "0.500");
  assert.equal(heatLevel(150, "%"), "1.000");
  assert.equal(heatLevel(50, "s"), "");
  assert.equal(heatLevel(null, "%"), "");
});
