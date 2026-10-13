// URL-hash routing. Every view and filter lives in the hash so any state can
// be bookmarked or shared:
//   #/                     criterion table
//   #/experiments          experiment list
//   #/exp?id=…&q=…         one experiment's samples
//   #/run?id=…&run=N       one sample

/** Split a location hash into its route name and query parameters. */
export function parseHash(hash) {
  const raw = hash.replace(/^#\/?/, "");
  const [route, query = ""] = raw.split("?");
  return { route: route || "", params: new URLSearchParams(query) };
}

/** Hash for a route; empty, null and undefined parameters are left out. */
export function hashFor(route, params = {}) {
  const query = new URLSearchParams(
    Object.entries(params).filter(([, value]) => value !== "" && value !== null && value !== undefined),
  ).toString();
  return `#/${route}${query ? `?${query}` : ""}`;
}

export function currentRoute() {
  return parseHash(window.location.hash);
}

/** Update query parameters in place, without adding a history entry. */
export function setParams(updates) {
  const { route, params } = currentRoute();
  for (const [key, value] of Object.entries(updates)) {
    if (value === "" || value === null || value === undefined) params.delete(key);
    else params.set(key, value);
  }
  window.history.replaceState(null, "", hashFor(route, Object.fromEntries(params)));
}

let renderCurrent = null;
let refreshTimer = null;

/** Register the function that renders the current route. */
export function onRoute(render) {
  renderCurrent = render;
}

/** Re-render the current route (after a filter change, for example). */
export function rerender() {
  window.clearTimeout(refreshTimer);
  return renderCurrent?.();
}

/** Re-render later unless the user has moved to another view meanwhile. */
export function refreshLater(ms) {
  const hash = window.location.hash;
  window.clearTimeout(refreshTimer);
  refreshTimer = window.setTimeout(() => {
    if (window.location.hash === hash) rerender();
  }, ms);
}

export function debounce(fn, ms) {
  let timer;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), ms);
  };
}
