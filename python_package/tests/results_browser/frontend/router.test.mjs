import assert from "node:assert/strict";
import { test } from "node:test";

import { hashFor, parseHash } from "../../../src/heams/results_browser/static/js/router.js";

test("hashFor drops empty parameters and encodes values", () => {
  assert.equal(hashFor("exp", { id: "run/exp 1", crit: "", page: null, sort: undefined }), "#/exp?id=run%2Fexp+1");
  assert.equal(hashFor("", {}), "#/");
});

test("parseHash round-trips hashFor", () => {
  const { route, params } = parseHash(hashFor("run", { id: "a/b", run: 3 }));
  assert.equal(route, "run");
  assert.equal(params.get("id"), "a/b");
  assert.equal(params.get("run"), "3");
  assert.equal(parseHash("").route, "");
  assert.equal(parseHash("#/").route, "");
});
