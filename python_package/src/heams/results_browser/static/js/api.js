// JSON API client with per-session caches for the expensive payloads.

let overview = null;
const experiments = new Map();

/** GET `/api/<path>`; rejects with the server's error message. */
export async function getJson(path) {
  const response = await fetch(`/api/${path}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error ?? response.statusText);
  return payload;
}

export function query(params) {
  return new URLSearchParams(
    Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== ""),
  ).toString();
}

/** The experiment list (cached; `refresh` rescans the results directory). */
export async function loadOverview(refresh = false) {
  if (!overview || refresh) {
    overview = await getJson(`overview${refresh ? "?refresh=1" : ""}`);
    experiments.clear();
  }
  return overview;
}

/** One experiment's metadata and sample summaries (cached). */
export async function loadExperiment(id) {
  if (!experiments.has(id)) experiments.set(id, await getJson(`experiment?${query({ id })}`));
  return experiments.get(id);
}

export const loadTable = (pick) => getJson(`table?${query({ pick })}`);
/** One sample; the last one is kept so filter changes re-render without a fetch. */
let lastRun = { path: null, payload: null };
export function loadRun(id, run, full) {
  const path = `run?${query({ id, run, full: full ? 1 : "" })}`;
  if (lastRun.path !== path) {
    lastRun = { path, payload: getJson(path) };
    lastRun.payload.catch(() => {
      if (lastRun.path === path) lastRun = { path: null, payload: null };
    });
  }
  return lastRun.payload;
}
export const loadTrace = (id, run) => getJson(`trace?${query({ id, run })}`);
export const loadTraceEvent = (id, run, event) => getJson(`trace_event?${query({ id, run, event })}`);
export const rawRunUrl = (id, run) => `/api/raw?${query({ id, run })}`;
export const reportUrl = (id) => `/api/report?${query({ id })}`;
