# BEAM benchmark

!!! success "Runnable via `heams run`"
    BEAM is available on the config-driven factory path
    (`type: beam` in `heams.benchmarks.factory`).

BEAM ([Tavakoli et al., 2025](https://arxiv.org/abs/2510.27246)) is a
long-context memory-ability suite: coherent chats at **100K / 500K / 1M / 10M**
tokens with ten probing abilities. LoCoMo and LongMemEval are largely
saturated; BEAM is the capacity stress-test.

This is **not** the Transparency benchmark. Transparency still downloads BEAM
as an *audit* adapter (`source.type: beam`) and trims chats to a handful of
turns. Native `type: beam` primes the **full** conversation once and grades
probing questions with rubric nuggets.

Implementation lives under `heams/benchmarks/beam/`
(`config`, `dataset`, `samples`, `benchmark`, `judge`, `metrics`, `report`).
Download/parse is shared with Transparency.

## Protocol

1. Download `Mohammadta/BEAM` (100K, 500K, 1M) or `Mohammadta/BEAM-10M`
2. Prime every chat turn as memory (user + assistant, timestamps, pair ingest)
3. Ask all probing questions (typically 20: 10 abilities × 2) as independent
   `UserStimulus` items (`reset_state()` before each probe; primed memory stays)
4. `beam_llm_judge` follows BEAM [§2.4 Evaluation](https://arxiv.org/abs/2510.27246):
   nine abilities score as the mean of LLM nugget grades `{0.0, 0.5, 1.0}`.
   `event_ordering` does **not** average nuggets; it aligns extracted events
   and scores `recall × (Kendall τ-b + 1) / 2` (presence × ordering). Empty
   extract is 0; a fully reversed complete list is 0; a correct-order subset
   of `k` of `n` events is `k/n` (half the events sits on `pass_threshold:
   0.5`). The paper reports the continuous score; `pass_threshold: 0.5` is
   harness binarization only.

Abilities: `preference_following`, `instruction_following`,
`information_extraction`, `multi_session_reasoning`, `knowledge_update`,
`summarization`, `temporal_reasoning`, `event_ordering`, `abstention`,
`contradiction_resolution`.

## Component configs

| Role | Bundled file |
|------|----------------|
| Agent (laptop) | `configs/agents/testing/gpt4o_mini_recall_only_tool_use.yaml` (`openai:gpt-4o-mini`) |
| Agent (campaign) | `configs/agents/campaigns/beam.yaml` (`openai:gpt-5.6-luna`, same settings as locomo/LME) |
| Memory (baseline) | `configs/memory_systems/testing/no_memory.yaml` |
| Memory (file-backed) | `configs/memory_systems/testing/basic_complete_memory_system.yaml` |
| Benchmark (laptop smoke) | `configs/benchmarks/testing/beam_smoke.yaml` (`100K`, `limit: 1`, `limit_stimuli: 1`) |
| Benchmark (campaign smoke) | `configs/benchmarks/smoke_campaign/beam_100k.yaml` (`limit: 1` conversation) |
| Benchmark (full splits) | `configs/benchmarks/full_campaign/beam_100k.yaml` … `beam_10m.yaml` |
| Run (laptop smoke) | `configs/runs/testing/beam_smoke.yaml` |

The memory-system smoke campaign (`configs/runs/smoke_campaign/*.yaml`) includes
the 100K slice. Full per-memory campaigns list all four token scales as
separate jobs; `in_context` omits `1M` and `10M` because those chats do not fit
a context window.

YAML owns `answer_prompt` and the evaluator block. There are no Python prompt
defaults. `local_path` points at a JSON list of conversations for tests.

## Run

From `python_package/`:

```sh
make run CONFIG=testing/beam_smoke OUTPUT_DIR=results/beam_smoke
# or
uv run heams run \
  --config configs/runs/testing/beam_smoke.yaml \
  --output-dir results/beam_smoke
```

Requires `OPENAI_API_KEY` for the default OpenAI agent and judge. First run
downloads the selected HuggingFace split (cached afterwards).

## Artefacts

After a successful run under the exact `output_dir` (for example
`results/beam_smoke/`):

- per-experiment `runs.jsonl`, `run_status.json`, `experiment_config.*.yaml`
- per-experiment `report.md` after `heams generate-report`
- `_rollup/beam/report.md`, only if you generate the optional rollup from Python:
  accuracy by memory backend, ability, and token scale

See [Understanding results](../results.md).

## Source and license

BEAM is by Tavakoli et al., 2025
([arXiv:2510.27246](https://arxiv.org/abs/2510.27246)). Its datasets,
[`Mohammadta/BEAM`](https://huggingface.co/datasets/Mohammadta/BEAM) and
[`Mohammadta/BEAM-10M`](https://huggingface.co/datasets/Mohammadta/BEAM-10M), are
licensed **CC BY-SA 4.0**. They are downloaded on first use and not
redistributed in this repository. See [Citations & credits](../citations.md) for BibTeX.
