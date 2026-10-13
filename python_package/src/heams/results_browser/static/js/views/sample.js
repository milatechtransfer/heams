// One sample: judge records, execution trace, transcript, metrics, ledger.

import { loadOverview, loadRun, rawRunUrl } from "../api.js";
import { html } from "../dom.js";
import { formatCount, formatPercent } from "../format.js";
import { mathEnabled, setMathEnabled } from "../math.js";
import { hashFor, rerender, setParams } from "../router.js";
import { byId, card, clampedText, experimentLabel, jsonBlock, keyValueGrid, options, setCrumbs, setRoot, show, textBlock } from "../ui.js";
import { showTrace } from "./trace.js";

function recordsTable(records, filter) {
  if (!records?.length) return html`<p class="muted">No judge records.</p>`;
  const rows = records.filter((record) => !filter || (filter === "fail" ? !record.passed : record.passed));
  return html`<div class="scroll"><table>
    <thead><tr><th>Result</th><th>Key</th><th>Tags</th><th class="col-expected">Expected</th><th class="col-wide">Observed</th><th class="col-wide">Judge reason</th></tr></thead>
    <tbody>${rows.map((record) =>
      record._truncated_items
        ? html`<tr><td colspan="6" class="muted">… ${record._truncated_items} more records (load untruncated)</td></tr>`
        : html`<tr>
          <td class="${record.passed ? "pass" : "fail"}">${record.passed ? "pass" : "fail"}${
            record.score !== undefined && record.score !== null && record.score !== Number(record.passed)
              ? html` <span class="muted">${record.score.toFixed(2)}</span>`
              : ""
          }</td>
          <td><code>${record.key}</code></td>
          <td>${Object.entries(record.tags ?? {}).map(([key, value]) => html`<span class="chip" title="${key}">${value}</span>`)}</td>
          <td>${clampedText(record.expected)}</td>
          <td>${clampedText(record.observed)}</td>
          <td>${clampedText(record.reason)}</td>
        </tr>`,
    )}</tbody>
  </table></div>`;
}

function transcript(turns) {
  if (!turns?.length) return html`<p class="muted">No transcript.</p>`;
  return turns.map((turn, index) =>
    turn._truncated_items
      ? html`<p class="muted">… ${turn._truncated_items} more turns (load untruncated)</p>`
      : html`<div class="turn" id="turn-${index}">
        <div class="muted">Turn ${index + 1}${turn.context_truncated ? html` · <span class="warn">context truncated</span>` : ""}</div>
        <div class="label">User</div>${textBlock(turn.question)}
        ${turn.target_answer !== undefined ? html`<div class="label">Target</div>${textBlock(turn.target_answer, "target")}` : ""}
        <div class="label">Agent</div>${textBlock(turn.answer, "answer")}
        ${turn.tool_results?.length ? html`<details><summary>${turn.tool_results.length} tool call(s)</summary>${turn.tool_results.map((call) =>
          call._truncated_items
            ? html`<p class="muted">… ${call._truncated_items} more</p>`
            : html`<div><b>${call.tool_name}</b>${textBlock(call.arguments)}<div class="label">Output</div>${textBlock(call.output)}</div>`,
        )}</details>` : ""}
      </div>`,
  );
}

function ledgerPreview(event) {
  const pick = (value, ...keys) => (value && typeof value === "object" ? keys.map((key) => value[key]).find(Boolean) : value);
  return String(pick(event.payload, "content", "query") ?? pick(event.result, "message") ?? "");
}

function ledgerPanel(ledger) {
  if (!ledger?.length) return "";
  return html`<details><summary>Memory ledger (${ledger.length} events shown)</summary><div class="scroll"><table>
    <thead><tr><th>#</th><th>Operation</th><th>Status</th><th>Component</th><th class="num">ms</th><th>Payload / result</th></tr></thead>
    <tbody>${ledger.map((event, index) =>
      event._truncated_items
        ? html`<tr><td colspan="6" class="muted">… ${event._truncated_items} more events (load untruncated)</td></tr>`
        : html`<tr>
          <td class="num">${index}</td><td>${event.operation_type}</td>
          <td class="${event.status === "error" ? "fail" : ""}">${event.status}</td><td>${event.component}</td>
          <td class="num">${event.duration_ms === undefined || event.duration_ms === null ? "–" : event.duration_ms.toFixed(1)}</td>
          <td><details><summary class="muted summary-plain">${ledgerPreview(event).slice(0, 140)}</summary>${jsonBlock({ payload: event.payload, result: event.result })}</details></td>
        </tr>`,
    )}</tbody>
  </table></div></details>`;
}

function summaryCards(run) {
  const audit = run.audit ?? {};
  const passed = (audit.passed_keys ?? []).length;
  const checks = passed + (audit.failed_keys ?? []).length;
  const metrics = run.system_metrics;
  return html`<div class="cards">
    ${audit.score_available === false
      ? card("Score unavailable", audit.score_unavailable_reason || "yes", "warn small")
      : card("Accuracy", formatPercent(audit.accuracy_score), audit.accuracy_score === 1 ? "pass" : audit.accuracy_score === 0 ? "fail" : "")}
    ${card("Successful", audit.is_successful ? html`<span class="pass">yes</span>` : html`<span class="fail">no</span>`)}
    ${card("Checks passed", `${passed} / ${checks}`)}
    ${metrics ? html`
      ${card("Duration", metrics.run_duration_ms ? `${(metrics.run_duration_ms / 1000).toFixed(1)} s` : "–")}
      ${card("LLM tokens", formatCount(metrics.llm_total_tokens))}
      ${card("Memory ops", metrics.memory_operation_count ?? 0)}` : ""}
  </div>`;
}

export async function renderSample(params) {
  const id = params.get("id");
  const number = Number(params.get("run"));
  const full = params.get("full") === "1";
  const overview = await loadOverview();
  setRoot(overview.results_dir);
  setCrumbs([
    [overview.title, "#/"],
    ["Experiments", "#/experiments"],
    [experimentLabel(overview, id), hashFor("exp", { id })],
    [`sample ${number}`],
  ]);
  show(html`<p class="muted">Loading sample…</p>`);
  const payload = await loadRun(id, number, full);
  const run = payload.run;
  const audit = run.audit ?? {};
  const recordFilter = params.get("rec") || "";
  const failed = (audit.report_records ?? []).filter((record) => record.passed === false).length;

  show(html`
    <div class="pager">
      <a class="button ${number <= 0 ? "invisible" : ""}" href="${hashFor("run", { id, run: number - 1 })}">‹ Prev sample</a>
      <span>Sample ${number + 1} of ${payload.count}</span>
      <a class="button ${number >= payload.count - 1 ? "invisible" : ""}" href="${hashFor("run", { id, run: number + 1 })}">Next sample ›</a>
      <span class="spacer"></span>
      <label title="Render LaTeX as math; off shows the source"><input type="checkbox" id="math" ${mathEnabled() ? "checked" : ""}> Typeset math</label>
      <a class="button" href="${rawRunUrl(id, number)}">Download raw JSON</a>
    </div>
    <h1>${run.scenario_id} <span class="muted">sample ${run.sample_index}</span></h1>
    ${payload.truncated ? html`<div class="notice">Long strings and lists were truncated for display.
      <a href="${hashFor("run", { id, run: number, full: 1 })}">Load untruncated</a> (may be slow for large runs).</div>` : ""}
    ${summaryCards(run)}
    <h2>Judge</h2>
    ${textBlock(audit.audit_notes, "expand")}
    <div class="controls">
      <select id="rec">${options([["", "All records"], ["fail", `Failed (${failed})`], ["pass", "Passed"]], recordFilter)}</select>
    </div>
    ${recordsTable(audit.report_records, recordFilter)}
    ${!audit.report_records && audit.failed_keys?.length ? html`<details open><summary>Failed keys</summary>${jsonBlock(audit.failed_keys)}</details>` : ""}
    <h2>Execution trace</h2>
    <div id="trace">${run.trace_id ? "" : html`<p class="muted">No execution trace for this sample; runs record one in <code>execution_traces.jsonl</code>.</p>`}</div>
    <h2>Transcript (${(run.transcript ?? []).length} turns)</h2>
    ${transcript(run.transcript)}
    <h2>Metrics</h2>
    ${run.system_metrics ? html`<div class="panel">${keyValueGrid(run.system_metrics)}</div>` : html`<p class="muted">No system metrics.</p>`}
    ${run.component_metrics ? html`<details><summary>Component metrics</summary>${jsonBlock(run.component_metrics)}</details>` : ""}
    ${run.component_performance_metrics ? html`<details><summary>Component performance metrics</summary>${jsonBlock(run.component_performance_metrics)}</details>` : ""}
    ${ledgerPanel(run.memory_ledger)}`);

  byId("math").addEventListener("change", (event) => {
    setMathEnabled(event.target.checked);
    rerender();
  });
  byId("rec").addEventListener("change", (event) => {
    setParams({ rec: event.target.value });
    rerender();
  });
  if (run.trace_id) showTrace(byId("trace"), id, number);
}
