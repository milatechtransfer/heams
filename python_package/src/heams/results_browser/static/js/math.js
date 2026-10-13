// Typeset LaTeX in text blocks, at no cost to pages without math.
//
// Blocks that may hold math carry `data-math`. After a view renders,
// `typesetMath` scans them for delimiters (`mathSegments`, pure and unit
// tested). Only when one holds math is Temml (vendored under static/vendor/)
// fetched, once per page; each block is then typeset when it nears the
// viewport. Temml builds MathML DOM nodes and sets styles through the CSSOM,
// so the page's Content-Security-Policy needs no inline styles.
//
// Timings are recorded as `performance` measures: `math:load` (fetching
// Temml) and `math:typeset` (one block, with its expression count).

const TEMML_DIR = "/static/vendor/temml-0.13.5/";
const SETTING_KEY = "heams.renderMath";
/** Inline math longer than this is more likely two prices than a formula. */
const MAX_INLINE_CHARS = 1000;
const CLOSERS = { "\\[": "\\]", "\\(": "\\)", $$: "$$", $: "$" };
const OPENER = /\\\[|\\\(|\$\$|\$/g;

/** Whether math is typeset (a per-viewer setting; on by default). */
export function mathEnabled() {
  try {
    return window.localStorage.getItem(SETTING_KEY) !== "off";
  } catch {
    return true;
  }
}

export function setMathEnabled(enabled) {
  try {
    window.localStorage.setItem(SETTING_KEY, enabled ? "on" : "off");
  } catch {
    // Storage blocked: the setting lasts until the page reloads.
  }
}

/**
 * Where a single `$` closes, following Pandoc's rule: the opening `$` is
 * followed by a non-space, the closing one preceded by a non-space and not
 * followed by a digit. "$5 and $10" stays text.
 */
function closeDollar(text, from) {
  if (/\s/.test(text[from] ?? " ")) return -1;
  let close = text.indexOf("$", from);
  while (close >= 0 && text[close - 1] === "\\") close = text.indexOf("$", close + 1);
  if (close < 0 || close - from > MAX_INLINE_CHARS) return -1;
  if (/\s/.test(text[close - 1]) || /\d/.test(text[close + 1] ?? "")) return -1;
  return text.slice(from, close).includes("\n\n") ? -1 : close;
}

/**
 * `text` split into `{text}` and `{tex, display}` segments. Delimiters:
 * `\[…\]` and `$$…$$` (display), `\(…\)` and `$…$` (inline).
 */
export function mathSegments(text) {
  const segments = [];
  let plain = 0;
  let at = 0;
  for (;;) {
    OPENER.lastIndex = at;
    const open = OPENER.exec(text);
    if (!open) break;
    const opener = open[0];
    const from = open.index + opener.length;
    const escaped = opener === "$" && text[open.index - 1] === "\\";
    const close = escaped ? -1 : opener === "$" ? closeDollar(text, from) : text.indexOf(CLOSERS[opener], from);
    const tex = close < 0 ? "" : text.slice(from, close);
    if (!tex.trim()) {
      at = from;
      continue;
    }
    if (open.index > plain) segments.push({ text: text.slice(plain, open.index) });
    segments.push({ tex, display: opener === "\\[" || opener === "$$" });
    plain = at = close + CLOSERS[opener].length;
  }
  if (plain < text.length) segments.push({ text: text.slice(plain) });
  return segments;
}

export function hasMath(text) {
  return mathSegments(text).some((segment) => segment.tex !== undefined);
}

let temml;

/** Fetch Temml and its stylesheet once; resolves to the `temml` global. */
function loadTemml() {
  temml ??= new Promise((resolve, reject) => {
    performance.mark("math:load-start");
    const style = Object.assign(document.createElement("link"), { rel: "stylesheet", href: `${TEMML_DIR}Temml-Local.css` });
    const script = Object.assign(document.createElement("script"), { src: `${TEMML_DIR}temml.min.js` });
    script.addEventListener("load", () => {
      performance.measure("math:load", "math:load-start");
      resolve(window.temml);
    });
    script.addEventListener("error", () => {
      temml = undefined;
      reject(new Error("could not load the math renderer"));
    });
    document.head.append(style, script);
  });
  return temml;
}

/** Replace a text node's math with typeset MathML; returns the expression count. */
function typesetText(node, renderer) {
  const segments = mathSegments(node.data);
  if (!segments.some((segment) => segment.tex !== undefined)) return 0;
  const fragment = document.createDocumentFragment();
  let count = 0;
  segments.forEach((segment, index) => {
    if (segment.tex === undefined) {
      // A display formula is a block of its own; drop the line breaks around it.
      let text = segment.text;
      if (segments[index - 1]?.display) text = text.replace(/^\n/, "");
      if (segments[index + 1]?.display) text = text.replace(/\n$/, "");
      fragment.append(text);
      return;
    }
    const holder = document.createElement("span");
    try {
      renderer.render(segment.tex, holder, { displayMode: segment.display, throwOnError: true });
      fragment.append(...holder.childNodes);
      count += 1;
    } catch {
      // Not valid LaTeX after all: keep the source as written.
      const [open, close] = segment.display ? ["\\[", "\\]"] : ["\\(", "\\)"];
      fragment.append(`${open}${segment.tex}${close}`);
    }
  });
  node.replaceWith(fragment);
  return count;
}

function typesetBlock(block, renderer) {
  if (!block.isConnected) return;
  const start = performance.now();
  const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  const expressions = nodes.reduce((total, node) => total + typesetText(node, renderer), 0);
  performance.measure("math:typeset", { start, detail: { expressions } });
}

let observer;

function whenVisible(block) {
  observer ??= new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        observer.unobserve(entry.target);
        loadTemml().then((renderer) => typesetBlock(entry.target, renderer), () => {});
      }
    },
    { rootMargin: "600px 0px" },
  );
  observer.observe(block);
}

/** Typeset the `[data-math]` blocks under `root` that hold math, lazily. */
export function typesetMath(root) {
  if (!mathEnabled() || typeof IntersectionObserver === "undefined") return;
  for (const block of root.querySelectorAll("[data-math]")) {
    if (hasMath(block.textContent)) whenVisible(block);
  }
}
