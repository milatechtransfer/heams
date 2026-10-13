// Lay out recorded values for reading without changing what they say.
// Benchmarks record text in many shapes: model answers stored escaped inside
// another string (a Python repr in a judge reason), JSON encoded as a string
// value, Python `str()` of a list or dict, nested tool payloads. These pure
// functions decode such text and lay structures out as indented
// `key: value` lines with real line breaks.

const ESCAPE = /\\(\\|n|t|r|'|"|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2})/g;
const DECODED = { "\\": "\\", n: "\n", t: "\t", r: "", "'": "'", '"': '"' };
const MAX_NESTED_JSON = 4;
// Terminal color and cursor codes some models write into their answers.
const ANSI_CODE = /\u001b\[[0-9;]*[A-Za-z]/g;

/** Text stored escaped: literal "\n" sequences and no real line break. */
export function looksEscaped(text) {
  return !text.includes("\n") && (text.match(/\\n/g) ?? []).length >= 2;
}

/** Decode the escapes of a Python repr or a JSON string body. */
export function unescapeText(text) {
  return text.replace(ESCAPE, (_, code) =>
    code.length > 1 ? String.fromCharCode(Number.parseInt(code.slice(1), 16)) : DECODED[code],
  );
}

const PYTHON_ATOM = /^(?:True|False|None|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)/;
const PYTHON_CONSTANTS = { True: true, False: false, None: null };

/**
 * A Python literal of lists, tuples, dicts, strings, numbers, booleans and
 * None, as `str()` writes it. Throws `SyntaxError` on anything else.
 */
export function parsePythonLiteral(text) {
  let at = 0;
  const fail = () => {
    throw new SyntaxError(`not a Python literal at ${at}`);
  };
  const space = () => {
    while (/\s/.test(text[at] ?? "")) at += 1;
  };
  const separator = (close) => {
    space();
    if (text[at] === ",") {
      at += 1;
      space();
    } else if (text[at] !== close) fail();
  };
  const string = (quote) => {
    const start = at + 1;
    for (at = start; text[at] !== quote; at += text[at] === "\\" ? 2 : 1) {
      if (at >= text.length) fail();
    }
    at += 1;
    return unescapeText(text.slice(start, at - 1));
  };
  const value = () => {
    space();
    const open = text[at];
    if (open === "[" || open === "(") {
      const close = open === "[" ? "]" : ")";
      const items = [];
      for (at += 1, space(); text[at] !== close; separator(close)) items.push(value());
      at += 1;
      return items;
    }
    if (open === "{") {
      const entries = {};
      for (at += 1, space(); text[at] !== "}"; separator("}")) {
        const key = value();
        space();
        if (text[at] !== ":") fail();
        at += 1;
        entries[String(key)] = value();
      }
      at += 1;
      return entries;
    }
    if (open === "'" || open === '"') return string(open);
    const atom = PYTHON_ATOM.exec(text.slice(at, at + 64));
    if (!atom) fail();
    at += atom[0].length;
    return atom[0] in PYTHON_CONSTANTS ? PYTHON_CONSTANTS[atom[0]] : Number(atom[0]);
  };
  const result = value();
  space();
  if (at !== text.length) fail();
  return result;
}

/** A JSON or Python-literal list or dict written as text, else undefined. */
function parseStructure(text) {
  const trimmed = text.trim();
  if (!/^[[{(]/.test(trimmed)) return undefined;
  for (const parse of [JSON.parse, parsePythonLiteral]) {
    try {
      const value = parse(trimmed);
      if (value && typeof value === "object") return value;
    } catch {
      // Not this syntax.
    }
  }
  return undefined;
}

/** `value` with JSON-encoded strings parsed and escaped strings decoded. */
export function expandValue(value, depth = 0) {
  if (typeof value === "string") {
    const parsed = depth < MAX_NESTED_JSON ? parseStructure(value) : undefined;
    if (parsed !== undefined) return expandValue(parsed, depth + 1);
    return (looksEscaped(value) ? unescapeText(value) : value).replace(ANSI_CODE, "");
  }
  if (Array.isArray(value)) return value.map((item) => expandValue(item, depth));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, expandValue(item, depth)]));
  }
  return value;
}

const isContainer = (value) => value !== null && typeof value === "object";
const isEmpty = (value) => (Array.isArray(value) ? value.length === 0 : Object.keys(value).length === 0);

function scalar(value) {
  return value === null || value === undefined ? "null" : String(value);
}

function indent(text, pad) {
  return text
    .split("\n")
    .map((line) => (line ? pad + line : line))
    .join("\n");
}

/** One `label` + value entry: inline when short, indented block otherwise. */
function entry(label, value, pad) {
  if (isContainer(value) && !isEmpty(value)) return `${pad}${label}\n${layout(value, `${pad}  `)}`;
  if (isContainer(value)) return `${pad}${label} ${Array.isArray(value) ? "[]" : "{}"}`;
  const text = scalar(value);
  if (!text.includes("\n")) return `${pad}${label} ${text}`;
  return `${pad}${label}\n${indent(text, `${pad}  `)}`;
}

/** Indented `key: value` / `- item` lines; multi-line strings keep their breaks. */
export function layout(value, pad = "") {
  if (!isContainer(value)) return indent(scalar(value), pad);
  if (Array.isArray(value)) {
    return value
      .map((item) =>
        item && item._truncated_items !== undefined
          ? `${pad}… ${item._truncated_items} more items`
          : entry("-", item, pad),
      )
      .join("\n");
  }
  return Object.entries(value)
    .map(([key, item]) => entry(`${key}:`, item, pad))
    .join("\n");
}

/** Display text for any recorded value. */
export function readableText(value) {
  const expanded = expandValue(value);
  return typeof expanded === "string" ? expanded : layout(expanded);
}
