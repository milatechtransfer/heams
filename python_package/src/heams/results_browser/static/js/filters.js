// Filtering and summarizing samples and trace events. Pure functions over the
// JSON the API returns, unit tested.

/**
 * Whether `text` contains every whitespace-separated term of `query`, in any
 * order, ignoring case ("letta beam" matches "BEAM 1M · letta_openai").
 */
export function matchesQuery(text, query) {
  const terms = (query ?? "").toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return true;
  const haystack = String(text ?? "").toLowerCase();
  return terms.every((term) => haystack.includes(term));
}

/**
 * A sample's score (0..1) for a table criterion, or `null` when the sample
 * has none. Scores come from the server with the same criterion rules as
 * the table, so a criterion opened from a table cell matches its value.
 */
export function criterionScore(run, criterion) {
  const score = run.scores?.[criterion];
  return score === undefined ? null : score;
}

/** `[passed, total]` judge records of a sample with a tag ("key=value"). */
export function tagCounts(run, tag) {
  return run.tags?.[tag] ?? null;
}

export const SAMPLE_STATUSES = [
  ["", "All samples"],
  ["pass", "Accuracy = 100%"],
  ["fail", "Accuracy < 100%"],
  ["zero", "Accuracy = 0%"],
  ["unscored", "Unscored"],
];

/** Whether a sample summary matches the sample-list filters. */
export function sampleMatches(run, { query = "", status = "", criterion = "", tag = "" }) {
  if (criterion && criterionScore(run, criterion) === null) return false;
  if (tag && !tagCounts(run, tag)) return false;
  const scored = run.score_available !== false;
  if (status === "pass" && !(scored && run.accuracy === 1)) return false;
  if (status === "fail" && !(scored && run.accuracy !== null && run.accuracy < 1)) return false;
  if (status === "zero" && !(scored && run.accuracy === 0)) return false;
  if (status === "unscored" && scored) return false;
  return matchesQuery(`${run.scenario_id} ${run.question} ${Object.keys(run.tags ?? {}).join(" ")}`, query);
}

/** Mean of the samples' scores for a criterion, with how many have one. */
export function criterionSummary(runs, criterion) {
  const scores = runs.map((run) => criterionScore(run, criterion)).filter((score) => score !== null);
  return { mean: scores.length ? scores.reduce((sum, score) => sum + score, 0) / scores.length : null, n: scores.length };
}

/** Judge records with a tag across samples: `[passed, total]`. */
export function tagSummary(runs, tag) {
  return runs.reduce(
    (sum, run) => {
      const counts = tagCounts(run, tag);
      return counts ? [sum[0] + counts[0], sum[1] + counts[1]] : sum;
    },
    [0, 0],
  );
}

/** Sort key spec "field" or "-field" applied to a copy of `runs`. */
export function sortSamples(runs, spec) {
  const direction = spec.startsWith("-") ? -1 : 1;
  const key = spec.replace(/^-/, "");
  return [...runs].sort((a, b) => {
    const left = a[key] ?? -Infinity;
    const right = b[key] ?? -Infinity;
    return (left > right ? 1 : left < right ? -1 : 0) * direction;
  });
}

/** Mean accuracy over scored samples only (unscored ones are excluded). */
export function meanAccuracy(runs) {
  const scored = runs.filter((run) => run.score_available !== false && run.accuracy !== null);
  if (!scored.length) return null;
  return scored.reduce((sum, run) => sum + run.accuracy, 0) / scored.length;
}

/**
 * Pass counts per judge-record tag: `Map(key => Map(value => [passed, total]))`,
 * keeping tags with 2 to 60 distinct values (a useful breakdown).
 */
export function tagBreakdown(runs) {
  const facets = new Map();
  for (const run of runs) {
    for (const [tag, [passed, total]] of Object.entries(run.tags ?? {})) {
      const split = tag.indexOf("=");
      const key = tag.slice(0, split);
      const value = tag.slice(split + 1);
      if (!facets.has(key)) facets.set(key, new Map());
      const counts = facets.get(key).get(value) ?? [0, 0];
      counts[0] += passed;
      counts[1] += total;
      facets.get(key).set(value, counts);
    }
  }
  return new Map([...facets].filter(([, values]) => values.size > 1 && values.size <= 60));
}

// -- execution traces -------------------------------------------------------

/**
 * Color slot of an event kind. Kinds and their order come from the trace
 * (declared by the framework); slots cycle through the palette.
 */
export function kindSlot(kind, kinds) {
  const index = Object.keys(kinds ?? {}).indexOf(kind);
  return `k${index < 0 ? 0 : index % 8}`;
}

export function isFailedEvent(event) {
  return event.status === "error" || Boolean(event.error);
}

/** Timeline filters from the URL parameters. */
export function traceFilters(params) {
  return {
    types: new Set((params.get("ttype") ?? "").split(",").filter(Boolean)),
    phase: params.get("tphase") ?? "",
    stimulus: params.get("tstim") ?? "",
    query: params.get("tq") ?? "",
    errorsOnly: params.get("terr") === "1",
  };
}

export function eventMatches(event, filters) {
  if (filters.types.size && !filters.types.has(event.type)) return false;
  if (filters.phase && event.phase !== filters.phase) return false;
  if (filters.stimulus !== "" && String(event.stimulus_index ?? "") !== filters.stimulus) return false;
  if (filters.errorsOnly && !isFailedEvent(event)) return false;
  const text = [
    event.name,
    event.type,
    event.content_preview,
    event.arguments_preview,
    event.result_preview,
    event.error,
  ].join(" ");
  return matchesQuery(text, filters.query);
}

function eventEnd(event) {
  return (event.start_offset_ms ?? 0) + (event.duration_ms ?? 0);
}

/**
 * Time axis of the timeline. With a filter on, it zooms to the matching
 * events' window; `end` is always the whole trace's duration.
 */
export function timelineWindow(events, matching) {
  const end = Math.max(1, ...events.map(eventEnd));
  const zoomed = matching.length > 0 && matching.length < events.length;
  const start = zoomed ? Math.min(...matching.map((event) => event.start_offset_ms ?? 0)) : 0;
  const stop = zoomed ? Math.max(...matching.map(eventEnd)) : end;
  return { start, stop, span: Math.max(stop - start, 1), end, zoomed };
}

/** Left offset and width (in %) of an event's bar within a window. */
export function barGeometry(event, window) {
  return {
    left: ((100 * ((event.start_offset_ms ?? 0) - window.start)) / window.span).toFixed(3),
    width: ((100 * (event.duration_ms ?? 0)) / window.span).toFixed(3),
  };
}

/** Counts shown above the timeline, by kind and by type. */
/**
 * The timeline group an event belongs to: its stimulus, or its phase for
 * events outside any stimulus (priming, manipulations).
 */
export function eventGroup(event) {
  const stimulus = event.stimulus_index ?? null;
  return stimulus === null
    ? { key: `phase:${event.phase}`, label: event.phase }
    : { key: `stimulus:${stimulus}`, label: `Stimulus ${stimulus}` };
}

export function traceSummary(events) {
  const byKind = new Map();
  const byType = new Map();
  let tokens = 0;
  let errors = 0;
  for (const event of events) {
    byKind.set(event.kind, (byKind.get(event.kind) ?? 0) + 1);
    const type = byType.get(event.type) ?? { kind: event.kind, count: 0, names: new Map() };
    type.count += 1;
    type.names.set(event.name, (type.names.get(event.name) ?? 0) + 1);
    byType.set(event.type, type);
    if (event.usage) tokens += (event.usage.input_tokens ?? 0) + (event.usage.output_tokens ?? 0);
    if (isFailedEvent(event)) errors += 1;
  }
  return { byKind, byType, tokens, errors };
}
