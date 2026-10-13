// Criterion table: memory systems as columns, one source run per benchmark.

import { loadOverview, loadTable } from "../api.js";
import { html } from "../dom.js";
import { matchesQuery } from "../filters.js";
import { formatValue, heatLevel, lowerIsBetter } from "../format.js";
import { debounce, hashFor, refreshLater, rerender, setParams } from "../router.js";
import { byId, options, overviewTabs, setCrumbs, setRoot, show, stateBadge } from "../ui.js";

const PICKS = [
  ["largest", "Largest run (most samples)"],
  ["latest", "Latest finished run"],
  ["latest_full", "Latest finished full-dataset run"],
];
let sourcesOpen = false;

const pairKey = (memory, benchmark) => `${memory}|${benchmark}`;
const hasLimit = (source) => source.limit !== null && source.limit !== undefined;

function shortRunName(source) {
  const parts = (source.run_dir || source.experiment).split("/");
  return parts[parts.length - 1];
}

function sourceTitle(source) {
  const scope = source.subset ? (hasLimit(source) ? ` (limit ${source.limit})` : " (subset)") : " (full dataset)";
  return `${source.run_dir || source.experiment}\nstarted ${source.started || "?"} · ${source.status} · ${source.samples ?? "?"} samples${scope}`;
}

function sourcesPanel(table) {
  const byPair = new Map(table.sources.map((source) => [pairKey(source.memory_system, source.benchmark), source]));
  const order = (benchmark) => table.rows.findIndex((row) => row.benchmark === benchmark);
  const benchmarks = [...new Set(table.sources.map((source) => source.benchmark))].sort((a, b) => order(a) - order(b));
  const rows = benchmarks.map((benchmark) => html`<tr><td class="group">${table.benchmark_labels[benchmark]}</td>${table.memory_systems.map((memory) => {
    const entry = byPair.get(pairKey(memory, benchmark));
    if (!entry) return html`<td class="muted">–</td>`;
    if (!entry.selected) return html`<td>${stateBadge(entry.state)} <span class="muted">no matching run</span></td>`;
    const selected = entry.selected;
    const others = entry.alternatives.length;
    return html`<td title="${sourceTitle(selected)}">
      <a href="${hashFor("exp", { id: selected.experiment })}">${shortRunName(selected)}</a>
      <span class="n">${(selected.started || "").slice(0, 10)} · n=${selected.samples ?? "?"}${hasLimit(selected) ? ` · limit ${selected.limit}` : ""}</span>
      ${entry.state !== "complete" ? stateBadge(entry.state) : ""}
      ${others ? html`<span class="n" title="${entry.alternatives.map(sourceTitle).join("\n\n")}">+${others} other run${others > 1 ? "s" : ""}</span>` : ""}
    </td>`;
  })}</tr>`);
  return html`<details class="sources" id="sources" ${sourcesOpen ? "open" : ""}>
    <summary>Sources: which run each cell comes from</summary>
    <div class="scroll"><table class="bbt">
      <thead><tr><th>Benchmark</th>${table.memory_systems.map((memory) => html`<th title="${memory}">${table.memory_labels[memory]}</th>`)}</tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  </details>`;
}

function valueCell(row, memory, source, memoryLabel) {
  const cell = row.cells[memory];
  if (!cell) {
    const state = source && source.state !== "complete" ? source.state : null;
    return html`<td class="num na">${state ? stateBadge(state) : "–"}</td>`;
  }
  const selected = source?.selected;
  const excluded = cell.excluded ? ` · ${cell.excluded} excl.` : "";
  const date = selected?.started ? ` · ${selected.started.slice(5, 10)}` : "";
  const title = `${memoryLabel} · ${row.benchmark_label} · ${row.criterion}\nn=${cell.n}${excluded}\n${
    selected ? sourceTitle(selected) : cell.experiment
  }`;
  const href = hashFor("exp", { id: cell.experiment, crit: row.criterion === "Overall" ? "" : row.criterion });
  return html`<td class="num cell ${cell.value === null ? "na" : ""}" data-heat="${heatLevel(cell.value, row.unit)}"
      title="${title}" data-href="${href}">
    <a class="cell-link ${cell.best ? "best" : ""}" href="${href}">${formatValue(cell.value, row.unit)}</a>
    <span class="n">n=${cell.n}${excluded}${date}${
      selected && selected.status !== "complete" ? html` · <span class="warn">${selected.status}</span>` : ""
    }</span>
  </td>`;
}

/** Mark the best value of each row (lowest for costs and error rates). */
function markBest(row, memories) {
  const values = memories.map((memory) => row.cells[memory]?.value).filter((value) => value !== null && value !== undefined);
  if (values.length < 2) return;
  const best = lowerIsBetter(row.unit, row.criterion) ? Math.min(...values) : Math.max(...values);
  for (const memory of memories) {
    const cell = row.cells[memory];
    if (cell) cell.best = cell.value === best;
  }
}

/** Section color slot: the section's position in the framework's order. */
function dimensionSlot(dimension, dimensions) {
  const index = dimensions.indexOf(dimension);
  return `dim-${index < 0 ? dimensions.length : index}`;
}

function tableBody(rows, memories, sources, dimensions, memoryLabels) {
  const body = [];
  let lastDimension = null;
  let lastBenchmark = null;
  for (const row of rows) {
    if (row.dimension !== lastDimension) {
      body.push(html`<tr class="dim ${dimensionSlot(row.dimension, dimensions)}"><td colspan="${memories.length + 2}">${row.dimension}</td></tr>`);
      lastDimension = row.dimension;
      lastBenchmark = null;
    }
    markBest(row, memories);
    const benchmarkCell = row.benchmark !== lastBenchmark ? row.benchmark_label : "";
    lastBenchmark = row.benchmark;
    body.push(html`<tr class="${row.criterion === "Overall" ? "overall" : ""}">
      <td class="group">${benchmarkCell}</td>
      <td>${row.criterion}${row.unit && row.unit !== "%" ? html` <span class="muted">(${row.unit})</span>` : ""}</td>
      ${memories.map((memory) => valueCell(row, memory, sources.get(pairKey(memory, row.benchmark)), memoryLabels[memory]))}
    </tr>`);
  }
  return body;
}

export async function renderTable(params) {
  const overview = await loadOverview();
  setRoot(overview.results_dir);
  setCrumbs([[overview.title]]);
  const pick = params.get("pick") || "largest";
  const table = await loadTable(pick);
  const memories = table.memory_systems;
  const filters = {
    dimension: params.get("dim") || "",
    benchmark: params.get("bench") || "",
    query: params.get("q") || "",
    overallOnly: params.get("overall") === "1",
  };
  const rows = table.rows.filter(
    (row) =>
      (!filters.dimension || row.dimension === filters.dimension) &&
      (!filters.benchmark || row.benchmark === filters.benchmark) &&
      (!filters.overallOnly || row.criterion === "Overall") &&
      matchesQuery(`${row.benchmark} ${row.benchmark_label} ${row.criterion}`, filters.query),
  );
  const sources = new Map(table.sources.map((source) => [pairKey(source.memory_system, source.benchmark), source]));
  const dimensions = [...new Set(table.rows.map((row) => row.dimension))];
  const benchmarks = [...new Set(table.rows.map((row) => row.benchmark))];
  const pending = table.pending.length;
  const body = tableBody(rows, memories, sources, table.dimensions, table.memory_labels);

  show(html`
    <h1>${overview.title}</h1>
    <div class="muted">${overview.experiments.length} experiments · ${memories.length} memory systems · <a href="#" id="refresh">rescan</a></div>
    ${overviewTabs("table")}
    <div class="controls">
      <label>Source run <select id="pick">${options(PICKS, pick)}</select></label>
      <select id="dim">${options([["", "All dimensions"], ...dimensions.map((d) => [d, d])], filters.dimension)}</select>
      <select id="bench">${options([["", "All benchmarks"], ...benchmarks.map((b) => [b, table.benchmark_labels[b]])], filters.benchmark)}</select>
      <input type="search" id="q" placeholder="Filter criteria…" value="${params.get("q") || ""}">
      <label><input type="checkbox" id="overall" ${filters.overallOnly ? "checked" : ""}> Overall rows only</label>
    </div>
    <p class="muted">Each memory system × benchmark uses one run, chosen by <b>Source run</b>; columns can mix runs.
      <b>Largest</b> counts every run, finished or not (unfinished ones are flagged); <b>Latest</b> only considers finished runs.
      Cells show sample count and run date; samples without a valid score are excluded (<i>excl.</i>).
      Hover for the run, click to open its samples. Underlined = best in row.</p>
    ${pending ? html`<div class="notice">Indexing ${pending} run${pending > 1 ? "s" : ""}; the table fills in automatically.
      First-time indexing of large <code>runs.jsonl</code> files can take a few minutes; it is cached afterwards.</div>` : ""}
    ${sourcesPanel(table)}
    <div class="scroll"><table class="bbt">
      <thead><tr><th>Benchmark</th><th>Criterion</th>${memories.map((memory) => html`<th class="num" title="${memory}">${table.memory_labels[memory]}</th>`)}</tr></thead>
      <tbody>${body.length ? body : html`<tr><td colspan="${memories.length + 2}" class="muted">${pending ? "Waiting for indexes…" : "No rows match."}</td></tr>`}</tbody>
    </table></div>`);

  const update = () => {
    setParams({
      pick: byId("pick").value === "largest" ? "" : byId("pick").value,
      dim: byId("dim").value,
      bench: byId("bench").value,
      q: byId("q").value,
      overall: byId("overall").checked ? "1" : "",
    });
    rerender();
  };
  for (const id of ["pick", "dim", "bench", "overall"]) byId(id).addEventListener("change", update);
  byId("q").addEventListener("input", debounce(update, 200));
  byId("sources").addEventListener("toggle", (event) => {
    sourcesOpen = event.target.open;
  });
  byId("refresh").addEventListener("click", async (event) => {
    event.preventDefault();
    await loadOverview(true);
    rerender();
  });
  // The value is a link (keyboard and screen readers); a click anywhere in
  // the cell follows it too.
  for (const cell of document.querySelectorAll("td.cell[data-href]")) {
    cell.addEventListener("click", (event) => {
      if (!event.target.closest("a")) window.location.hash = cell.dataset.href;
    });
  }
  if (pending) refreshLater(2000);
}
