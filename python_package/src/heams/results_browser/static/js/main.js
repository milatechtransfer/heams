// Entry point: dispatch the URL hash to a view and keep focus while typing.

import { html } from "./dom.js";
import { currentRoute, onRoute, rerender } from "./router.js";
import { bindExpanders, show } from "./ui.js";
import { renderExperiment } from "./views/experiment.js";
import { renderExperiments } from "./views/experiments.js";
import { renderSample } from "./views/sample.js";
import { renderTable } from "./views/table.js";

const VIEWS = {
  "": renderTable,
  experiments: renderExperiments,
  exp: renderExperiment,
  run: renderSample,
};

async function renderRoute() {
  const { route, params } = currentRoute();
  // Views re-render on filter changes; keep the focused input and caret.
  const focused = document.activeElement?.id;
  const caret = document.activeElement?.selectionStart;
  try {
    await (VIEWS[route] ?? renderTable)(params);
  } catch (error) {
    show(html`<p class="error">${error.message}</p>`);
  }
  const input = focused ? document.getElementById(focused) : null;
  if (input) {
    input.focus();
    if (caret !== undefined && caret !== null && input.setSelectionRange) input.setSelectionRange(caret, caret);
  }
}

bindExpanders();
onRoute(renderRoute);
window.addEventListener("hashchange", () => {
  rerender();
  window.scrollTo(0, 0);
});
rerender();
