// Execution trace timeline of one sample: a waterfall of every framework and
// agent event, with filters, and full event detail loaded on demand.

import { loadTrace, loadTraceEvent } from "../api.js";
import { html, mount } from "../dom.js";
import { formatCount, formatDuration } from "../format.js";
import {
  barGeometry,
  eventGroup,
  eventMatches,
  isFailedEvent,
  kindSlot,
  timelineWindow,
  traceFilters,
  traceSummary,
} from "../filters.js";
import { currentRoute, debounce, setParams } from "../router.js";
import { card, jsonBlock, keyValueGrid, textBlock } from "../ui.js";

const PAGE = 500;
const DETAIL_FIELDS = ["phase", "type", "name", "status", "source", "start_offset_ms", "duration_ms", "step_index", "stimulus_index"];

/** Load a sample's trace into `container` and render it. */
export async function showTrace(container, experimentId, runIndex) {
  mount(container, html`<p class="muted">Loading execution trace…</p>`);
  try {
    const trace = await loadTrace(experimentId, runIndex);
    const state = { container, experimentId, runIndex, trace, shown: PAGE };
    render(state);
  } catch (error) {
    mount(container, html`<p class="muted">${error.message}</p>`);
  }
}

function typeBadge(type, kind, kinds) {
  return html`<span class="tb" title="${kinds[kind] ?? kind}"><i class="${kindSlot(kind, kinds)}"></i>${type}</span>`;
}

function eventRow(event, axis, kinds) {
  const failed = isFailedEvent(event);
  const preview = event.error || event.content_preview || event.arguments_preview || event.result_preview || "";
  const bar = barGeometry(event, axis);
  const number = (event.sequence_index ?? event.i) + 1;
  return html`<tr class="trow ${failed ? "terror" : ""}" data-i="${event.i}">
      <td class="num"><button type="button" class="trow-toggle" aria-expanded="false" aria-controls="tdetail-${event.i}"
        aria-label="Details of event ${number}">${number}</button></td>
      <td class="num">+${formatDuration(event.start_offset_ms)}</td>
      <td class="num">${formatDuration(event.duration_ms)}</td>
      <td>${typeBadge(event.type, event.kind, kinds)}</td>
      <td><b>${event.name}</b>${event.step_index !== undefined && event.step_index !== null ? html` <span class="muted">step ${event.step_index}</span>` : ""}</td>
      <td class="${failed ? "fail" : "muted"}">${event.status ?? ""}</td>
      <td class="wfcell"><div class="wf"><span class="${kindSlot(event.kind, kinds)}" data-left="${bar.left}" data-width="${bar.width}"></span></div></td>
      <td>
        <div class="clamp short">${preview}</div>
        ${event.arguments_preview && event.arguments_preview !== preview ? html`<div class="muted hint">args: ${event.arguments_preview}</div>` : ""}
        <span class="muted hint">${event.phase}${event.input_messages ? ` · ${event.input_messages} input msg` : ""}</span>
      </td>
    </tr>
    <tr class="tdetail" id="tdetail-${event.i}" hidden><td colspan="8"></td></tr>`;
}

function eventRows(events, axis, kinds) {
  const rows = [];
  let lastGroup;
  for (const event of events) {
    const group = eventGroup(event);
    if (group.key !== lastGroup) {
      rows.push(html`<tr class="tgroup"><td colspan="8">${group.label}</td></tr>`);
      lastGroup = group.key;
    }
    rows.push(eventRow(event, axis, kinds));
  }
  return rows;
}

function render(state) {
  const { trace, container } = state;
  const params = currentRoute().params;
  const filters = traceFilters(params);
  const events = trace.events;
  const summary = traceSummary(events);
  const kinds = trace.kinds ?? {};
  const matching = events.filter((event) => eventMatches(event, filters));
  const shown = matching.slice(0, state.shown);
  const axis = timelineWindow(events, matching);
  const phases = [...new Set(events.map((event) => event.phase))];
  const stimuli = [...new Set(events.map((event) => event.stimulus_index).filter((value) => value !== null && value !== undefined))].sort((a, b) => a - b);
  const remaining = matching.length - shown.length;

  mount(container, html`
    <div class="cards">
      ${card("Outcome", trace.outcome, trace.outcome === "failed" ? "fail" : "pass")}
      ${card("Events", formatCount(events.length))}
      ${card("Wall time", formatDuration(axis.end))}
      ${Object.entries(kinds)
        .filter(([kind]) => summary.byKind.has(kind))
        .map(([kind, label]) => card(label, formatCount(summary.byKind.get(kind))))}
      ${card("LLM tokens", formatCount(summary.tokens))}
      ${card("Errors", summary.errors, summary.errors ? "fail" : "")}
    </div>
    ${trace.error ? html`<div class="notice error">${trace.error}</div>` : ""}
    <div class="controls">
      ${[...summary.byType].map(([type, { kind, count, names }]) => html`<a href="#" class="chip tfilter ${filters.types.has(type) ? "on" : ""}"
          data-type="${type}" title="${[...names].map(([name, n]) => `${name} × ${n}`).join("\n")}">${typeBadge(type, kind, kinds)} ${count}</a>`)}
    </div>
    <div class="controls">
      <select id="tphase">${[["", "All phases"], ...phases.map((phase) => [phase, phase])].map(([value, label]) => html`<option value="${value}" ${value === filters.phase ? "selected" : ""}>${label}</option>`)}</select>
      ${stimuli.length ? html`<select id="tstim">${[["", "All stimuli"], ...stimuli.map((value) => [String(value), `Stimulus ${value}`])].map(([value, label]) => html`<option value="${value}" ${value === filters.stimulus ? "selected" : ""}>${label}</option>`)}</select>` : ""}
      <input type="search" id="tq" placeholder="Search names, content, arguments…" value="${params.get("tq") || ""}">
      <label><input type="checkbox" id="terr" ${filters.errorsOnly ? "checked" : ""}> Errors only</label>
      <span class="muted">${formatCount(matching.length)} of ${formatCount(events.length)} events · click a row for details</span>
    </div>
    <div class="scroll"><table class="trace">
      <thead><tr>
        <th class="num">#</th><th class="num">Start</th><th class="num">Duration</th><th>Type</th><th>Name</th><th>Status</th>
        <th class="col-timeline">Timeline (${axis.zoomed ? `+${formatDuration(axis.start)} – +${formatDuration(axis.stop)}` : `0 – ${formatDuration(axis.end)}`})</th>
        <th>Content</th>
      </tr></thead>
      <tbody>${shown.length ? eventRows(shown, axis, kinds) : html`<tr><td colspan="8" class="muted">No events match.</td></tr>`}</tbody>
    </table></div>
    ${remaining > 0 ? html`<p><button id="tmore">Show ${Math.min(PAGE, remaining)} more of ${remaining}</button></p>` : ""}`);

  bindControls(state, filters);
}

function bindControls(state, filters) {
  const { container } = state;
  const update = (changes) => {
    setParams(changes);
    render(state);
  };
  for (const chip of container.querySelectorAll(".tfilter")) {
    chip.addEventListener("click", (event) => {
      event.preventDefault();
      const types = new Set(filters.types);
      if (types.has(chip.dataset.type)) types.delete(chip.dataset.type);
      else types.add(chip.dataset.type);
      update({ ttype: [...types].join(",") });
    });
  }
  container.querySelector("#tphase").addEventListener("change", (event) => update({ tphase: event.target.value }));
  container.querySelector("#tstim")?.addEventListener("change", (event) => update({ tstim: event.target.value }));
  container.querySelector("#terr").addEventListener("change", (event) => update({ terr: event.target.checked ? "1" : "" }));
  const search = container.querySelector("#tq");
  search.addEventListener(
    "input",
    debounce(() => {
      const caret = search.selectionStart;
      update({ tq: search.value });
      const input = container.querySelector("#tq");
      input.focus();
      input.setSelectionRange(caret, caret);
    }, 250),
  );
  container.querySelector("#tmore")?.addEventListener("click", () => {
    state.shown += PAGE;
    render(state);
  });
  // Each row's number is a button (keyboard and screen readers); a click
  // anywhere in the row toggles the same details.
  for (const row of container.querySelectorAll("tr.trow")) {
    row.addEventListener("click", () => toggleDetail(state, Number(row.dataset.i)));
  }
}

function messageBlock(message, open) {
  const content = typeof message.content === "string" ? message.content : JSON.stringify(message.content ?? message, null, 2);
  return html`<details ${open ? "open" : ""}>
    <summary>${message.role || "message"}${message.name ? ` · ${message.name}` : ""} <span class="muted summary-plain">${formatCount((content || "").length)} chars</span></summary>
    ${textBlock(content)}
    ${message.tool_calls ? html`<div class="label">Tool calls</div>${jsonBlock(message.tool_calls)}` : ""}
  </details>`;
}

function eventDetail(event, truncated) {
  const meta = Object.fromEntries(DETAIL_FIELDS.filter((key) => event[key] !== undefined && event[key] !== null).map((key) => [key, event[key]]));
  const messages = Array.isArray(event.input_messages) ? event.input_messages : null;
  return html`<div class="tevent">
    ${truncated ? html`<div class="notice">Long fields are shortened for display.</div>` : ""}
    <div class="kv">${Object.entries(meta).map(([key, value]) => html`<div>${key}</div><div>${key.endsWith("_ms") ? formatDuration(value) : value}</div>`)}</div>
    ${event.usage ? html`<div class="label">Usage</div>${keyValueGrid(event.usage)}` : ""}
    ${event.error ? html`<div class="label">Error</div>${textBlock(event.error, "error")}` : ""}
    ${event.content !== undefined ? html`<div class="label">Content</div>${textBlock(event.content)}` : ""}
    ${event.arguments && Object.keys(event.arguments).length ? html`<div class="label">Arguments</div>${textBlock(event.arguments)}` : ""}
    ${event.result !== undefined && event.result !== null ? html`<div class="label">Result</div>${textBlock(event.result)}` : ""}
    ${messages ? html`<div class="label">Input messages (${messages.length})</div>${messages.map((message, index) =>
      message._truncated_items
        ? html`<p class="muted">… ${message._truncated_items} more messages</p>`
        : messageBlock(message, index === messages.length - 1),
    )}` : ""}
  </div>`;
}

async function toggleDetail(state, position) {
  const row = state.container.querySelector(`#tdetail-${position}`);
  if (!row) return;
  const toggle = state.container.querySelector(`tr.trow[data-i="${position}"] .trow-toggle`);
  toggle?.setAttribute("aria-expanded", String(row.hidden));
  if (!row.hidden) {
    row.hidden = true;
    return;
  }
  row.hidden = false;
  const cell = row.firstElementChild;
  mount(cell, html`<p class="muted">Loading event…</p>`);
  try {
    const { event, truncated } = await loadTraceEvent(state.experimentId, state.runIndex, position);
    mount(cell, eventDetail(event, truncated));
  } catch (error) {
    mount(cell, html`<p class="error">${error.message}</p>`);
  }
}
