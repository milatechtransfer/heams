import assert from "node:assert/strict";
import { test } from "node:test";

import {
  barGeometry,
  criterionScore,
  criterionSummary,
  eventGroup,
  eventMatches,
  kindSlot,
  matchesQuery,
  meanAccuracy,
  sampleMatches,
  sortSamples,
  tagBreakdown,
  tagCounts,
  tagSummary,
  timelineWindow,
  traceFilters,
  traceSummary,
} from "../../../src/heams/results_browser/static/js/filters.js";

const run = (fields) => ({ scenario_id: "s", question: "q", tags: {}, accuracy: 1, score_available: true, ...fields });

test("search matches every term in any order, ignoring case", () => {
  assert.ok(matchesQuery("BEAM 1M · letta_openai", "letta beam"));
  assert.ok(matchesQuery("anything", "  "));
  assert.ok(!matchesQuery("BEAM 1M · mem0", "letta beam"));
});

test("criteria use the server's per-sample scores", () => {
  const runs = [run({ scores: { "erase / pre": 0.5 } }), run({ scores: { "erase / pre": 1 } }), run({ scores: {} })];
  assert.equal(criterionScore(runs[0], "erase / pre"), 0.5);
  assert.equal(criterionScore(runs[2], "erase / pre"), null);
  assert.deepEqual(criterionSummary(runs, "erase / pre"), { mean: 0.75, n: 2 });
  assert.deepEqual(criterionSummary(runs, "missing"), { mean: null, n: 0 });
  assert.ok(sampleMatches(runs[0], { criterion: "erase / pre" }));
  assert.ok(!sampleMatches(runs[2], { criterion: "erase / pre" }));
});

test("tags filter on exact key=value records", () => {
  const runs = [run({ tags: { "difficulty=hard": [1, 2] } }), run({ tags: { "difficulty=easy": [1, 1] } })];
  assert.deepEqual(tagCounts(runs[0], "difficulty=hard"), [1, 2]);
  assert.deepEqual(tagSummary(runs, "difficulty=hard"), [1, 2]);
  assert.deepEqual(tagSummary(runs, "difficulty=none"), [0, 0]);
  assert.ok(!sampleMatches(runs[1], { tag: "difficulty=hard" }));
});

test("sample filters treat unscored samples separately", () => {
  const passed = run({ accuracy: 1 });
  const zero = run({ accuracy: 0 });
  const unscored = run({ accuracy: 0, score_available: false });
  assert.ok(sampleMatches(passed, { status: "pass" }));
  assert.ok(sampleMatches(zero, { status: "zero" }));
  assert.ok(!sampleMatches(unscored, { status: "zero" }));
  assert.ok(!sampleMatches(unscored, { status: "fail" }));
  assert.ok(sampleMatches(unscored, { status: "unscored" }));
  assert.ok(sampleMatches(run({ question: "Where is Paris?" }), { query: "paris" }));
});

test("mean accuracy ignores unscored samples", () => {
  assert.equal(meanAccuracy([run({ accuracy: 1 }), run({ accuracy: 0, score_available: false })]), 1);
  assert.equal(meanAccuracy([run({ score_available: false })]), null);
});

test("samples sort ascending and descending with missing values last", () => {
  const runs = [run({ i: 0, tokens: 5 }), run({ i: 1, tokens: null }), run({ i: 2, tokens: 9 })];
  assert.deepEqual(sortSamples(runs, "tokens").map((r) => r.i), [1, 0, 2]);
  assert.deepEqual(sortSamples(runs, "-tokens").map((r) => r.i), [2, 0, 1]);
});

test("tag breakdown keeps tags with between 2 and 60 values", () => {
  const breakdown = tagBreakdown([
    run({ tags: { "category=a": [1, 1], "only=x": [1, 1] } }),
    run({ tags: { "category=b": [0, 2], "only=x": [0, 1] } }),
  ]);
  assert.deepEqual([...breakdown.keys()], ["category"]);
  assert.deepEqual(breakdown.get("category").get("b"), [0, 2]);
});

// Types and kinds are opaque strings declared by the framework.
const events = [
  { i: 0, type: "setup_a", kind: "setup", phase: "priming", name: "prime", start_offset_ms: 0, duration_ms: 10 },
  { i: 1, type: "call_b", kind: "llm", phase: "execution", name: "gpt", start_offset_ms: 100, duration_ms: 400, stimulus_index: 0, usage: { input_tokens: 10, output_tokens: 2 } },
  { i: 2, type: "call_c", kind: "tool", phase: "execution", name: "recall", start_offset_ms: 600, duration_ms: 400, stimulus_index: 0, status: "error", error: "timeout" },
];

test("kinds get color slots by their declared position", () => {
  const kinds = { setup: "Setup", llm: "LLM calls", tool: "Tool calls" };
  assert.equal(kindSlot("setup", kinds), "k0");
  assert.equal(kindSlot("tool", kinds), "k2");
  assert.equal(kindSlot("unknown", kinds), "k0");
  assert.equal(kindSlot("x", Object.fromEntries([...Array(10)].map((_, i) => [i === 9 ? "x" : `k${i}`, ""]))), "k1");
});

test("trace filters combine type, phase, stimulus, text and errors", () => {
  const filters = (query) => traceFilters(new URLSearchParams(query));
  const match = (query) => events.filter((event) => eventMatches(event, filters(query))).map((event) => event.i);
  assert.deepEqual(match(""), [0, 1, 2]);
  assert.deepEqual(match("ttype=call_b,call_c"), [1, 2]);
  assert.deepEqual(match("tphase=priming"), [0]);
  assert.deepEqual(match("tstim=0"), [1, 2]);
  assert.deepEqual(match("terr=1"), [2]);
  assert.deepEqual(match("tq=RECALL"), [2]);
});

test("the timeline zooms to the filtered window", () => {
  const whole = timelineWindow(events, events);
  assert.deepEqual([whole.start, whole.stop, whole.end, whole.zoomed], [0, 1000, 1000, false]);
  const zoomed = timelineWindow(events, events.slice(1));
  assert.deepEqual([zoomed.start, zoomed.stop, zoomed.end, zoomed.zoomed], [100, 1000, 1000, true]);
  assert.deepEqual(barGeometry(events[2], zoomed), { left: "55.556", width: "44.444" });
});

test("trace summary counts kinds, types with their names, tokens and errors", () => {
  const summary = traceSummary(events);
  assert.equal(summary.byKind.get("llm"), 1);
  assert.equal(summary.byType.get("call_c").kind, "tool");
  assert.equal(summary.byType.get("call_c").names.get("recall"), 1);
  assert.equal(summary.tokens, 12);
  assert.equal(summary.errors, 1);
});

test("events group by stimulus, or by phase outside stimuli", () => {
  const groups = [
    { phase: "priming" },
    { phase: "manipulation", stimulus_index: null },
    { phase: "execution", stimulus_index: 0 },
    { phase: "execution", stimulus_index: 1 },
  ].map((event) => eventGroup(event));
  assert.deepEqual(groups.map((group) => group.label), ["priming", "manipulation", "Stimulus 0", "Stimulus 1"]);
  assert.equal(new Set(groups.map((group) => group.key)).size, 4);
});
