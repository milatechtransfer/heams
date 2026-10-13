// Shared page chrome and small markup components used by several views.

import { html, mount } from "./dom.js";
import { formatCount } from "./format.js";
import { readableText } from "./readable.js";

export const app = document.getElementById("app");

/** Render the main content area. */
export function show(markup) {
  mount(app, markup);
}

/** Breadcrumbs from `[label, href?]` pairs; the last one is usually plain. */
export function setCrumbs(items) {
  mount(
    document.getElementById("crumbs"),
    items.map(([label, href], index) => html`${index ? html`<span class="sep">/</span>` : ""}${
      href ? html`<a href="${href}">${label}</a>` : html`<span>${label}</span>`
    }`),
  );
}

export function setRoot(path) {
  document.getElementById("root").textContent = path;
}

export function experimentLabel(overview, id) {
  const experiment = overview.experiments.find((item) => item.id === id);
  return experiment ? `${experiment.benchmark_label} · ${experiment.memory_label}` : id;
}

export function stateBadge(state) {
  return html`<span class="state ${state}">${state}</span>`;
}

export function card(label, value, valueClass = "") {
  return html`<div class="card"><div class="k">${label}</div><div class="v ${valueClass}">${value}</div></div>`;
}

export function overviewTabs(active) {
  return html`<div class="tabs">
    <a href="#/" class="${active === "table" ? "active" : ""}">Criterion table</a>
    <a href="#/experiments" class="${active === "experiments" ? "active" : ""}">Experiments</a>
  </div>`;
}

export function options(choices, selected) {
  return choices.map(([value, label]) => html`<option value="${value}" ${value === selected ? "selected" : ""}>${label}</option>`);
}

export function jsonBlock(value) {
  return html`<pre>${JSON.stringify(value, null, 2)}</pre>`;
}

/** Any recorded value laid out for reading (see readable.js); "—" when empty. */
export function textBlock(value, className = "") {
  if (value === null || value === undefined || value === "") return html`<span class="muted">—</span>`;
  return html`<pre class="${className}" data-math>${readableText(value)}</pre>`;
}

/** Text clamped to a few lines; a click expands or collapses it. */
export function clampedText(value, className = "") {
  if (value === null || value === undefined || value === "") return "";
  return html`<div class="clamp expand ${className}" title="Click to expand" data-math>${readableText(value)}</div>`;
}

/** Toggle `.expand` blocks on click, unless the click selected text. */
export function bindExpanders() {
  document.addEventListener("click", (event) => {
    const block = event.target.closest?.(".expand");
    if (!block || event.target.closest("a") || String(window.getSelection?.() ?? "")) return;
    block.classList.toggle("open");
  });
}

export function keyValueGrid(entries) {
  return html`<div class="kv">${Object.entries(entries ?? {}).map(
    ([key, value]) => html`<div>${key}</div><div>${
      typeof value === "number" ? (Number.isInteger(value) ? formatCount(value) : value.toFixed(4)) : JSON.stringify(value)
    }</div>`,
  )}</div>`;
}

/** Make `<tr data-href>` rows navigate on click (links inside keep working). */
export function bindRowLinks(root = app) {
  for (const row of root.querySelectorAll("tr[data-href]")) {
    row.addEventListener("click", (event) => {
      if (event.target.closest("a")) return;
      window.location.hash = row.dataset.href;
    });
  }
}

export function byId(id) {
  return document.getElementById(id);
}
