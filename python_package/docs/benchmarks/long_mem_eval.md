# LongMemEval benchmark

!!! success "Runnable via `heams run`"
    LongMemEval runs on the config-driven factory path
    (`type: long_mem_eval` in `heams.benchmarks.factory`).

LongMemEval loads the cleaned Hugging Face `longmemeval_s_cleaned.json` dataset by
default (downloaded on first use into `python_package/datasets/long_mem_eval/`),
primes each question's timestamped chat history into memory, and grades the single
per-scenario answer with an LLM-as-judge evaluator. Scenarios carry no
environment config.

The LongMemEval integration uses upstream-inspired generation and grading prompts
adapted to this package's memory-tool architecture: the agent is instructed to
recall relevant chat history before answering, while the judge uses task-specific
rubrics for factual, temporal, knowledge-update, preference, and abstention
questions.

Supported cleaned splits include `longmemeval_oracle.json`,
`longmemeval_s_cleaned.json`, and `longmemeval_m_cleaned.json` under
`datasets/long_mem_eval/`. The bundled configs default `examples_dir` to that
directory and therefore materialize the S split when no file is already present.
Point `examples_dir` at an M or oracle filename to select another split.

The judge model defaults to `openai:gpt-4o-mini` (configured in
`LongMemEvalLlmJudgeConfig`). The smoke run uses the
`testing/gpt4o_mini_recall_only_tool_use.yaml` agent preset; the campaign runs use
`campaigns/long_mem_eval.yaml` (`openai:gpt-5.6-luna`). Both call OpenAI, so
`OPENAI_API_KEY` is required.

## Bundled configurations

| Configuration | File | Behavior |
|---------------|------|----------|
| Benchmark (full) | `configs/benchmarks/full_campaign/long_mem_eval.yaml` | Default S split with no sample limit |
| Benchmark (smoke) | `configs/benchmarks/testing/long_mem_eval_smoke.yaml` | Same workload with `limit: 1` |
| Run (smoke) | `configs/runs/testing/long_mem_eval_smoke.yaml` | One-sample comparison of `no_memory` and `basic_complete_memory_system` |

Run the smoke campaign from `python_package/`:

```sh
make run CONFIG=testing/long_mem_eval_smoke OUTPUT_DIR=results/long_mem_eval_smoke
# or
uv run heams run --config configs/runs/testing/long_mem_eval_smoke.yaml --output-dir results/long_mem_eval_smoke
```

Full-scale LongMemEval runs are the `long_mem_eval.yaml` entries in each
per-memory `configs/runs/*_all_benchmarks/` campaign matrix.

Each experiment writes a resolved configuration, `runs.jsonl`, `run_status.json`,
and a memory workspace under its configured results directory. LongMemEval's
records are tagged by question type and abstention. Run `heams generate-report`
for a `report.md` per experiment, or inspect `runs.jsonl`; LongMemEval has no
per-benchmark rollup.

## Source and license

LongMemEval is by Wu et al., 2024
([arXiv:2410.10813](https://arxiv.org/abs/2410.10813)). Its cleaned dataset,
[`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned),
is licensed **MIT**. It is downloaded on first use and not redistributed in this
repository. See [Citations & credits](../citations.md) for BibTeX.

## Related docs

- [CLI](../cli.md) — config-driven runs
- [Understanding results](../results.md) — run artefacts
- [LoCoMo](locomo.md) — the other long-conversation QA benchmark
