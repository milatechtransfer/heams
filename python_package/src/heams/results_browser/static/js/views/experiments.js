// Experiment list: every experiment found under the results directory.

import { loadOverview } from "../api.js";
import { html } from "../dom.js";
import { matchesQuery } from "../filters.js";
import { formatBytes } from "../format.js";
import { debounce, hashFor, rerender, setParams } from "../router.js";
import { bindRowLinks, byId, overviewTabs, setCrumbs, setRoot, show, stateBadge } from "../ui.js";

export async function renderExperiments(params) {
  const overview = await loadOverview();
  setRoot(overview.results_dir);
  setCrumbs([[overview.title, "#/"], ["Experiments"]]);
  const query = params.get("q") || "";
  const experiments = overview.experiments
    .filter((e) =>
      matchesQuery(
        `${e.id} ${e.benchmark} ${e.benchmark_label} ${e.memory_system} ${e.memory_label} ${e.status} ${e.started}`,
        query,
      ),
    )
    .sort((a, b) => (b.run_id || "").localeCompare(a.run_id || "") || a.id.localeCompare(b.id));

  show(html`
    <h1>${overview.title}</h1>
    ${overviewTabs("experiments")}
    <div class="controls">
      <input type="search" id="q" placeholder="Filter by path, benchmark, memory, status, date…" value="${params.get("q") || ""}">
      <span class="muted">${experiments.length} of ${overview.experiments.length}, newest first</span>
    </div>
    <div class="scroll"><table>
      <thead><tr>
        <th>Started</th><th>Benchmark</th><th>Memory system</th><th>Status</th>
        <th class="num">Samples</th><th>Scope</th><th>Model</th><th>Experiment</th><th class="num">runs.jsonl</th>
      </tr></thead>
      <tbody>${experiments.map((e) => html`
        <tr class="clickable" data-href="${hashFor("exp", { id: e.id })}">
          <td class="num">${e.started || "–"}</td>
          <td>${e.benchmark_label}</td>
          <td title="${e.memory_system}">${e.memory_label}</td>
          <td>${stateBadge(e.status)}</td>
          <td class="num">${e.samples ?? "–"}</td>
          <td>${e.subset ? (e.limit !== null && e.limit !== undefined ? `limit ${e.limit}` : "subset") : html`<span class="muted">full</span>`}</td>
          <td class="muted">${e.agent_model}</td>
          <td><a href="${hashFor("exp", { id: e.id })}">${e.id}</a></td>
          <td class="num">${formatBytes(e.runs_bytes)}</td>
        </tr>`)}</tbody>
    </table></div>`);

  byId("q").addEventListener(
    "input",
    debounce(() => {
      setParams({ q: byId("q").value });
      rerender();
    }, 200),
  );
  bindRowLinks();
}
