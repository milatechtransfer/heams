// HTML building with escaping by construction.
//
// Views build markup with the `html` tagged template. Every interpolated value
// is escaped unless it is itself `Markup` (the result of another `html` call),
// so data from results files can never inject markup. Arrays are rendered
// item by item; `null`, `undefined` and `false` render as nothing.

import { typesetMath } from "./math.js";

const ESCAPES = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

/** Markup that is already safe to insert. */
export class Markup {
  constructor(value) {
    this.value = value;
  }

  toString() {
    return this.value;
  }
}

export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ESCAPES[char]);
}

function render(value) {
  if (value instanceof Markup) return value.value;
  if (Array.isArray(value)) return value.map(render).join("");
  if (value === null || value === undefined || value === false) return "";
  return escapeHtml(value);
}

/** Tagged template: `html\`<b>${name}</b>\`` escapes `name`. */
export function html(strings, ...values) {
  let out = strings[0];
  values.forEach((value, index) => {
    out += render(value) + strings[index + 1];
  });
  return new Markup(out);
}

/**
 * Replace an element's content, then apply the dynamic styles requested
 * through data attributes. Style attributes in markup would need a
 * Content-Security-Policy exception; setting them through the CSSOM does not.
 *
 *   data-heat="0..1"           background shade for percentages
 *   data-left / data-width     absolute position of a timeline bar, in %
 *   data-fill="0..100"         width of a progress bar, in %
 */
export function mount(element, markup) {
  element.innerHTML = render(markup);
  applyDynamicStyles(element);
  typesetMath(element);
}

export function applyDynamicStyles(root) {
  for (const el of root.querySelectorAll("[data-heat]")) {
    if (el.dataset.heat === "") continue;
    el.style.background = `rgba(var(--heat), ${(Number(el.dataset.heat) * 0.35).toFixed(3)})`;
  }
  for (const el of root.querySelectorAll("[data-left]")) {
    el.style.left = `${el.dataset.left}%`;
    el.style.width = `${el.dataset.width}%`;
  }
  for (const el of root.querySelectorAll("[data-fill]")) {
    el.style.width = `${el.dataset.fill}%`;
  }
}
