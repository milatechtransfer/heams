// Number formatting. Pure functions, unit tested. Display names of
// benchmarks, memory systems and sections come from the API, which reads
// them from the framework's declarations.

/** Units where a smaller value is better (highlighted as best in a row). */
const LOWER_IS_BETTER_UNITS = new Set(["s", "tokens", "operations", "USD"]);

export function lowerIsBetter(unit, criterion) {
  return LOWER_IS_BETTER_UNITS.has(unit) || /error/i.test(criterion);
}

/** A table value in its unit; `null` (no scored samples) reads "N/A". */
export function formatValue(value, unit) {
  if (value === null || value === undefined) return "N/A";
  if (unit === "%") return `${value.toFixed(1)}%`;
  if (unit === "s") return `${value.toFixed(2)} s`;
  if (unit === "tokens" || unit === "operations") return Math.round(value).toLocaleString("en-US");
  if (unit === "USD") return `$${value.toFixed(4)}`;
  if (Math.abs(value) >= 1000) return Math.round(value).toLocaleString("en-US");
  return Number.isInteger(value) ? String(value) : value.toFixed(3);
}

/** A ratio in [0, 1] as a percentage. */
export function formatPercent(ratio) {
  return ratio === null || ratio === undefined ? "–" : `${(ratio * 100).toFixed(1)}%`;
}

export function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

/** A duration in milliseconds, at the precision a timeline needs. */
export function formatDuration(ms) {
  if (ms === null || ms === undefined) return "–";
  if (ms < 1000) return `${ms.toFixed(ms < 10 ? 1 : 0)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)} s`;
  return `${Math.floor(ms / 60000)}m ${((ms % 60000) / 1000).toFixed(0)}s`;
}

export function formatCount(value) {
  return Number(value ?? 0).toLocaleString("en-US");
}

/** Background shade (0..1) for a percentage cell, or "" for other units. */
export function heatLevel(value, unit) {
  if (unit !== "%" || value === null || value === undefined) return "";
  return Math.max(0, Math.min(1, value / 100)).toFixed(3);
}
