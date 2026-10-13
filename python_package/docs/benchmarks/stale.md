# STALE benchmark

STALE ([icedreamc/STALE](https://github.com/icedreamc/STALE)) tests whether a
memory-augmented agent recognizes when a stored memory has become **stale**. Each
source sample describes a user state that changed from `M_old` to `M_new` inside a
primed conversation (`haystack_session`), then probes the agent along three
dimensions:

| Dimension | Name | Passes when the response… |
|---|---|---|
| `dim1` | Explicit Probing (State Resolution) | treats the old fact as possibly no longer valid |
| `dim2` | Adversarial Robustness (Premise Resistance) | detects a false premise built on the old state |
| `dim3` | Implicit Task (Implicit Policy Adaptation) | takes actions that adhere to the **new** state |

Each source record becomes **one scenario with three stimuli** (dim1/dim2/dim3):
the `haystack_session` is primed into memory once, and the agent answers all three
probes in a single run so the all-in-one judge can score them together. The agent
recalls the primed history via the memory tool and answers from the user's
most up-to-date state.

!!! note "Contamination trade-off"
    Because the three probes share one conversation, an earlier probe can inform a
    later one (e.g. after dim1 surfaces the new state, the dim2 false premise is
    easier to catch). This differs from STALE's upstream setup, which runs the
    target model **independently** per dimension and only combines them at judge
    time. We adopt STALE's all-in-one judge (below) and accept the shared-context
    trade-off. Set `agent_conversation: independent` to reset the agent's
    messages before each probe, as upstream does; the judge is unchanged.

## Dataset

The published benchmark lives on Hugging Face:
[`STALEproj/STALE`](https://huggingface.co/datasets/STALEproj/STALE) — a single
file `T1_T2_400_FULL.json` (~306 MB, **400 records**). It is CC-BY-4.0; distractor
sessions derive from LongMemEval (MIT), which retains its original attribution.

The `source:` block in the benchmark config controls where the corpus comes from:

- **Default** — download `filename` from the Hugging Face dataset `repo_id` via
  `hf_hub_download` (cached once under `~/.cache/huggingface`).
- **`local_path`** — read a local `*_MAIN.json` instead of downloading.
- **`auto_download: false`** — make a missing dataset a hard error (no network).

The full run is large (400 records = 400 scenarios per experiment, each running
three probes + one judge call), so set a `limit:` for a cheaper real-data slice.

Schema per record: `uid`, `M_old`, `M_new`, `explanation`,
`type` (`T1`/`T2`, carried into report tags as `conflict_type`),
`probing_queries.{dim1_query, dim2_query, dim3_query}`,
`haystack_session` (a **list of sessions**, each a list of `{role, content}`
turns), `timestamps` (session-level, one per session), and
`relevant_session_index` (the sessions holding the `M_old` / `M_new`
observations, surfaced as retrieval evidence). The loader also accepts a simpler
flat `haystack_session` and a `conflict_type` key; both shapes load.

## Evaluation

Scoring uses the `stale_llm_judge` evaluator (`heams/benchmarks/stale/judge.py`). It
mirrors STALE's **all-in-one** judge: a **single** call scores all three responses
against `M_old` / `M_new` / `explanation` and returns JSON
`{dim1_eval, dim2_eval, dim3_eval}` (each with `reasoning` and `pass`), parsed via
`LlmJudge(output_type=…)`. Per-scenario accuracy is the passed-dimension rate;
report records are tagged by `dimension` and `conflict_type`, which the per-tag
tables in `report.md` and the optional rollup group by.

The judge prompts are **authored in the benchmark YAML**, not hardcoded — the
`evaluator:` block sets `judge_system_prompt` (verbatim STALE rubric) and
`judge_user_template` (the ground-truth context plus the three question/response
pairs). The judge model defaults to `openai:gpt-4o-mini`, so `OPENAI_API_KEY` is
required for real runs.

## Running

```bash
uv run heams run --config configs/runs/<your_run>.yaml --output-dir results/<your_run>
```

Reference the benchmark from an experiment via
`benchmark_config_file: testing/stale_smoke.yaml` (or `full_campaign/stale.yaml`),
or start from the shipped `configs/runs/testing/stale_smoke.yaml`.

## Source and license

STALE is by Chao et al., 2026
([arXiv:2605.06527](https://arxiv.org/abs/2605.06527)). The dataset is licensed
**CC BY 4.0** and the [code](https://github.com/icedreamc/STALE) **MIT**. The
dataset is downloaded on first use; only one record is committed, as a test
fixture. See [Citations & credits](../citations.md) for BibTeX.

## Related docs

- [CLI](../cli.md) — config-driven runs
- [Transparency](transparency.md) — closest typed-path memory-audit benchmark
- [Adding a benchmark](../adding-benchmark.md) — factory migration checklist
