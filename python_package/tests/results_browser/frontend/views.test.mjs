// Render every view against a stub page and stub API responses. Catches
// errors that only happen at render time, such as a reference to a variable
// that is not in scope.

import assert from "node:assert/strict";
import { before, test } from "node:test";

class StubElement {
  constructor(id) {
    this.id = id;
    this.innerHTML = "";
    this.textContent = "";
    this.value = "";
    this.checked = false;
    this.hidden = false;
    this.dataset = {};
    this.style = {};
  }

  addEventListener() {}
  querySelectorAll() {
    return [];
  }
  querySelector(selector) {
    // Controls a view just rendered are looked up by id.
    return selector.startsWith("#") ? element(selector.slice(1)) : null;
  }
  focus() {}
  setSelectionRange() {}
}

const elements = new Map();
const element = (id) => {
  if (!elements.has(id)) elements.set(id, new StubElement(id));
  return elements.get(id);
};

const experiment = {
  id: "run/exp",
  name: "exp",
  benchmark: "bench:v1",
  benchmark_type: "bench",
  benchmark_label: "Declared Bench v1",
  variant: "v1",
  memory_system: "mem_a",
  memory_label: "Declared Memory",
  agent_model: "model",
  status: "complete",
  run_dir: "run",
  run_id: "20260101-000000-abcd1234",
  started: "2026-01-01 00:00",
  samples: 1,
  planned_samples: 1,
  limit: null,
  scope: "full",
  subset: false,
  runs_bytes: 10,
  has_submetrics: false,
  has_traces: true,
};
const summary = {
  i: 0,
  scenario_id: "s-1",
  trace_id: "t-1",
  sample_index: 0,
  accuracy: 1,
  success: true,
  score_available: true,
  passed: 1,
  failed: 0,
  records: 1,
  turns: 1,
  question: "Q?",
  duration_s: 1.5,
  tokens: 42,
  tags: { "category=temporal": [1, 1] },
  scores: { temporal: 1 },
};
const RESPONSES = {
  overview: { results_dir: "/results", title: "results", experiments: [experiment] },
  table: {
    pick: "largest",
    dimensions: ["Performance", "Privacy", "Efficiency", "Other"],
    memory_systems: ["mem_a"],
    memory_labels: { mem_a: "Declared Memory" },
    benchmark_labels: { "bench:v1": "Declared Bench v1" },
    rows: [
      {
        dimension: "Performance",
        benchmark: "bench:v1",
        benchmark_label: "Declared Bench v1",
        criterion: "Overall",
        unit: "%",
        cells: { mem_a: { value: 75, n: 4, excluded: 1, experiment: "run/exp" } },
      },
    ],
    sources: [
      {
        memory_system: "mem_a",
        benchmark: "bench:v1",
        state: "complete",
        selected: { experiment: "run/exp", run_dir: "run", started: "2026-01-01 00:00", samples: 4, limit: null, subset: false, status: "complete" },
        alternatives: [],
      },
    ],
    pending: [],
  },
  experiment: { experiment, config: "type: bench", run_status: "{}", has_report: false, headline: [], runs: [summary] },
  run: {
    count: 1,
    truncated: false,
    run: {
      scenario_id: "s-1",
      sample_index: 0,
      trace_id: "t-1",
      audit: { accuracy_score: 1, is_successful: true, passed_keys: ["q0"], failed_keys: [], report_records: [{ key: "q0", passed: true, tags: { category: "temporal" } }] },
      transcript: [{ question: "Q?", answer: "A.", tool_results: [{ tool_name: "recall", arguments: {}, output: "x" }] }],
      system_metrics: { run_duration_ms: 1500, llm_total_tokens: 42 },
    },
  },
  trace: {
    trace_id: "t-1",
    outcome: "completed",
    kinds: { setup: "Setup", llm: "LLM calls" },
    events: [
      { i: 0, type: "kind_a", kind: "setup", phase: "priming", name: "prime", start_offset_ms: 0, duration_ms: 2 },
      { i: 1, type: "kind_b", kind: "llm", phase: "execution", name: "model", start_offset_ms: 5, duration_ms: 10, stimulus_index: 0, usage: { input_tokens: 3 } },
    ],
  },
};

let views;

before(async () => {
  globalThis.document = { getElementById: element, activeElement: null, querySelectorAll: () => [] };
  globalThis.window = {
    location: { hash: "#/" },
    history: { replaceState() {} },
    setTimeout: () => 0,
    clearTimeout() {},
    addEventListener() {},
    scrollTo() {},
  };
  globalThis.fetch = async (url) => {
    const route = new URL(url, "http://stub").pathname.replace("/api/", "");
    return { ok: true, json: async () => RESPONSES[route] };
  };
  views = {
    table: await import("../../../src/heams/results_browser/static/js/views/table.js"),
    experiments: await import("../../../src/heams/results_browser/static/js/views/experiments.js"),
    experiment: await import("../../../src/heams/results_browser/static/js/views/experiment.js"),
    sample: await import("../../../src/heams/results_browser/static/js/views/sample.js"),
  };
});

const app = () => element("app").innerHTML;

test("the criterion table renders declared labels", async () => {
  await views.table.renderTable(new URLSearchParams());
  assert.match(app(), /Declared Bench v1/);
  assert.match(app(), /Declared Memory/);
  assert.match(app(), /75\.0%/);
  assert.match(app(), /class="dim dim-0"/);
  // Each value is a real link, reachable from the keyboard.
  assert.match(app(), /<a class="cell-link [^"]*" href="#\/exp\?id=run%2Fexp">75\.0%<\/a>/);
});

test("the experiment list renders", async () => {
  await views.experiments.renderExperiments(new URLSearchParams());
  assert.match(app(), /Declared Bench v1/);
});

test("an experiment renders its samples", async () => {
  await views.experiment.renderExperiment(new URLSearchParams("id=run/exp"));
  assert.match(app(), /Declared Bench v1 · Declared Memory/);
  assert.match(app(), /s-1/);
});

test("a criterion opened from a table cell scores like the table", async () => {
  await views.experiment.renderExperiment(new URLSearchParams("id=run/exp&crit=temporal"));
  assert.match(app(), /Criterion &quot;temporal&quot;/);
  assert.match(app(), /100\.0% · n=1/);
  assert.doesNotMatch(app(), /NaN/);
});

test("a sample renders with its trace timeline", async () => {
  await views.sample.renderSample(new URLSearchParams("id=run/exp&run=0"));
  assert.match(app(), /Execution trace/);
  await new Promise((resolve) => setImmediate(resolve));
  const trace = element("trace").innerHTML;
  assert.match(trace, /LLM calls/);
  assert.match(trace, /class="k1"/);
  // Rows open with a focusable button; a phase change starts a new group.
  assert.match(trace, /<button type="button" class="trow-toggle" aria-expanded="false" aria-controls="tdetail-0"/);
  assert.match(trace, /<tr class="tgroup"><td colspan="8">priming<\/td><\/tr>/);
  assert.match(trace, /<tr class="tgroup"><td colspan="8">Stimulus 0<\/td><\/tr>/);
});
