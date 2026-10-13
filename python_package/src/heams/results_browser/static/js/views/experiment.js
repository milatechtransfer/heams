// One experiment: metadata, per-tag pass rates, and its samples.

import { loadExperiment, loadOverview, reportUrl } from "../api.js";
import { html } from "../dom.js";
import { formatCount, formatPercent, formatValue } from "../format.js";
import {
  criterionScore,
  criterionSummary,
  meanAccuracy,
  SAMPLE_STATUSES,
  sampleMatches,
  sortSamples,
  tagBreakdown,
  tagCounts,
  tagSummary,
} from "../filters.js";
import { debounce, hashFor, rerender, setParams } from "../router.js";
import { bindRowLinks, byId, card, experimentLabel, options, setCrumbs, setRoot, show, stateBadge } from "../ui.js";

const PAGE_SIZE = 100;

function breakdownPanel(runs, id) {
  const breakdown = tagBreakdown(runs);
  if (!breakdown.size) return "";
  return html`<details ${breakdown.size <= 4 ? "open" : ""}>
    <summary>Judge-record breakdown by tag</summary>
    <div class="facets">${[...breakdown].map(([key, values]) => html`<div class="panel"><b>${key}</b><table>${[...values]
      .sort((a, b) => b[1][1] - a[1][1])
      .map(([value, [passed, total]]) => html`<tr>
        <td><a href="${hashFor("exp", { id, tag: `${key}=${value}` })}">${value}</a></td>
        <td class="num">${passed}/${total}</td>
        <td class="num">${formatPercent(passed / total)}</td>
        <td class="bar-cell"><div class="bar"><span data-fill="${((100 * passed) / total).toFixed(1)}"></span></div></td>
      </tr>`)}</table></div>`)}</div>
  </details>`;
}

function headlinePanel(rows) {
  const shown = rows.filter((row) => row.dimension !== "Efficiency" || row.criterion === "Overall");
  if (!shown.length) return "";
  return html`<details><summary>Headline aggregates (submetrics.csv)</summary><div class="scroll"><table>
    <thead><tr><th>Dimension</th><th>Criterion</th><th>Metric</th><th class="num">Value</th><th class="num">n</th><th class="num">Excluded</th></tr></thead>
    <tbody>${shown.map((row) => html`<tr>
      <td>${row.dimension}</td><td>${row.criterion}</td><td>${row.metric_name}</td>
      <td class="num">${formatValue(row.metric_value === "" ? null : Number(row.metric_value), row.unit)}</td>
      <td class="num">${row.sample_count}</td><td class="num">${row.excluded_count}</td>
    </tr>`)}</tbody>
  </table></div></details>`;
}

/** The selected criterion's score, or the selected tag's record counts. */
function focusCell(run, criterion, tag) {
  if (criterion) return formatPercent(criterionScore(run, criterion));
  const counts = tagCounts(run, tag);
  return counts ? `${counts[0]}/${counts[1]}` : "–";
}

function sampleRow(run, id, criterion, tag) {
  const unscored = run.score_available === false;
  const accuracyClass = unscored ? "warn" : run.accuracy === 1 ? "pass" : run.accuracy === 0 ? "fail" : "";
  const href = hashFor("run", { id, run: run.i });
  return html`<tr class="clickable" data-href="${href}">
    <td class="num"><a href="${href}">${run.i}</a></td>
    <td>${run.scenario_id}${run.sample_index ? html` <span class="muted">#${run.sample_index}</span>` : ""}</td>
    <td class="num ${accuracyClass}">${unscored ? "unscored" : formatPercent(run.accuracy)}</td>
    ${criterion || tag ? html`<td class="num">${focusCell(run, criterion, tag)}</td>` : ""}
    <td class="num"><span class="pass">${run.passed}</span> / <span class="fail">${run.failed}</span></td>
    <td class="num">${run.turns}</td>
    <td class="num">${run.duration_s === null ? "–" : `${run.duration_s.toFixed(1)} s`}</td>
    <td class="num">${run.tokens === null ? "–" : formatCount(run.tokens)}</td>
    <td><div class="clamp short" data-math>${run.question}</div></td>
  </tr>`;
}

export async function renderExperiment(params) {
  const id = params.get("id");
  const overview = await loadOverview();
  setRoot(overview.results_dir);
  setCrumbs([[overview.title, "#/"], ["Experiments", "#/experiments"], [experimentLabel(overview, id)]]);
  show(html`<p class="muted">Indexing runs.jsonl… (large files take a while the first time; the index is cached)</p>`);
  const detail = await loadExperiment(id);
  const experiment = detail.experiment;
  const runs = detail.runs;
  const criterion = params.get("crit") || "";
  const tag = criterion ? "" : params.get("tag") || "";
  const sort = params.get("sort") || "i";
  const page = Number(params.get("page") || 0);
  const status = params.get("status") || "";

  const filtered = sortSamples(
    runs.filter((run) => sampleMatches(run, { query: params.get("q") || "", status, criterion, tag })),
    sort,
  );
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const shown = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const unscored = runs.filter((run) => run.score_available === false).length;
  // Same rules as the table, so a criterion opened from a cell matches it.
  const criterionStats = criterion ? criterionSummary(filtered, criterion) : null;
  const tagStats = tag ? tagSummary(filtered, tag) : null;

  const sortHeader = (label, field, className = "") => {
    const next = sort === field ? `-${field}` : field;
    const arrow = sort === field ? " ▲" : sort === `-${field}` ? " ▼" : "";
    return html`<th class="${className}"><a href="#" data-sort="${next}">${label}${arrow}</a></th>`;
  };

  show(html`
    <h1>${experiment.benchmark_label} · ${experiment.memory_label}</h1>
    <div class="muted">${experiment.id}</div>
    <div class="cards">
      ${card("Status", stateBadge(experiment.status))}
      ${card("Samples", runs.length)}
      ${card("Mean accuracy", formatPercent(meanAccuracy(runs)))}
      ${unscored ? card("Unscored samples", unscored, "warn") : ""}
      ${card("Fully successful", `${runs.filter((run) => run.success).length} / ${runs.length}`)}
      ${card("Agent model", experiment.agent_model, "small")}
      ${criterionStats ? card(`Criterion "${criterion}"`, `${formatPercent(criterionStats.mean)} · n=${criterionStats.n}`) : ""}
      ${tagStats ? card(`Tag "${tag}"`, tagStats[1] ? `${tagStats[0]}/${tagStats[1]} · ${formatPercent(tagStats[0] / tagStats[1])}` : "–") : ""}
    </div>
    ${headlinePanel(detail.headline)}
    ${detail.config ? html`<details><summary>Experiment config</summary><pre>${detail.config}</pre></details>` : ""}
    ${detail.run_status ? html`<details><summary>run_status.json</summary><pre>${detail.run_status}</pre></details>` : ""}
    ${detail.has_report ? html`<p><a href="${reportUrl(id)}" target="_blank" rel="noopener">Open report.md</a></p>` : ""}
    ${breakdownPanel(runs, id)}
    <h2>Samples</h2>
    <div class="controls">
      <input type="search" id="q" placeholder="Search scenario, question, tags…" value="${params.get("q") || ""}">
      <select id="status">${options(SAMPLE_STATUSES, status)}</select>
      ${criterion ? html`<span class="chip">criterion: ${criterion} <a href="#" id="clear-focus" aria-label="Clear criterion">✕</a></span>` : ""}
      ${tag ? html`<span class="chip">tag: ${tag} <a href="#" id="clear-focus" aria-label="Clear tag">✕</a></span>` : ""}
      <span class="muted">${filtered.length} of ${runs.length}</span>
    </div>
    <div class="scroll"><table>
      <thead><tr>
        ${sortHeader("#", "i", "num")}${sortHeader("Scenario", "scenario_id")}${sortHeader("Accuracy", "accuracy", "num")}
        ${criterion ? html`<th class="num">Criterion score</th>` : tag ? html`<th class="num">Tag records</th>` : ""}
        ${sortHeader("Pass / fail", "failed", "num")}${sortHeader("Turns", "turns", "num")}
        ${sortHeader("Duration", "duration_s", "num")}${sortHeader("Tokens", "tokens", "num")}<th>First question</th>
      </tr></thead>
      <tbody>${shown.map((run) => sampleRow(run, id, criterion, tag))}</tbody>
    </table></div>
    <div class="pager">
      <button id="prev" ${page <= 0 ? "disabled" : ""}>‹ Prev</button>
      <span>Page ${page + 1} / ${pages}</span>
      <button id="next" ${page >= pages - 1 ? "disabled" : ""}>Next ›</button>
    </div>`);

  const update = (extra = {}) => {
    setParams({ q: byId("q").value, status: byId("status").value, page: "", ...extra });
    rerender();
  };
  byId("q").addEventListener("input", debounce(() => update(), 250));
  byId("status").addEventListener("change", () => update());
  byId("clear-focus")?.addEventListener("click", (event) => {
    event.preventDefault();
    update({ crit: "", tag: "" });
  });
  byId("prev").addEventListener("click", () => update({ page: page - 1 || "" }));
  byId("next").addEventListener("click", () => update({ page: page + 1 }));
  for (const link of document.querySelectorAll("a[data-sort]")) {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      update({ sort: link.dataset.sort === "i" ? "" : link.dataset.sort, page: page || "" });
    });
  }
  bindRowLinks();
}
