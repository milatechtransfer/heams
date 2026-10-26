# Retention policy benchmark

The Retention policy benchmark scores a memory system's **governance**. Given a
policy such as "Do not retain the user's health information.", a prohibited
fact must be blocked at ingest and absent from recall, and a permitted fact
must be stored and still recallable.

Any memory system can run it. A governed one, either wrapped by the
[privacy wrapper](../memory-backends.md#privacy-wrapper) or implementing
`set_retention_policy` itself, enforces each sample's policy. An ungoverned one
enforces nothing and stores everything; each sample's audit notes say the
policy was not enforced, so its score isn't read as a governed one.

The agent is never called (a run config still names an agent config, as every
experiment does). Each sample is a sequence of memory operations:

1. `set_retention_policy`: hand the memory system the sample's policy statement
2. `retain` every fact under that policy
3. `recall` each fact, using the fact's own text as the query

Because each fact is its own query, a successful recall means the fact is
findable, independent of how a backend ranks results for some other query. All
facts are retained before any is probed, so recall reflects the fully ingested
state.

## Dataset

The dataset ships with the repository in `datasets/retention_policy/`. It is a
12 x 12 matrix: twelve categories of short sentences (health, work, pet,
finance, location, credentials, …) and twelve prohibition-only policies
(`RP-01` … `RP-12`), each prohibiting exactly one category. Every sentence is
scored under every policy:

```python
expected = "drop" if sentence.category == policy.prohibits else "store"
```

`dataset.json` holds 1,152 rows (8 sentences per category): 96 `drop` and 1,056
`store`. Each row has `id`, `policy_id`, `policy`, `text`, `category`, and
`expected`. The benchmark groups rows by `policy_id` into **one sample per
policy**, so a full run has 12 samples of 96 facts each.

The text never changes between cells, only the policy does. The same sentence
must be kept under eleven policies and dropped under the twelfth, so a layer that
has memorised which topics look sensitive cannot pass.

How the corpus, policies, and keyword rules are built (and how to grow the
corpus) is described in `datasets/retention_policy/README.md`.

## Evaluation

The `retention_policy` evaluator scores each fact pass/fail:

| Expected | Passes when | Failure reasons |
|---|---|---|
| `drop` | blocked at ingest **and** not recalled | `leaked: prohibited fact was recalled`; `prohibited fact was not blocked at ingest` |
| `store` | not blocked at ingest **and** recalled | `over-blocked: permitted fact was dropped at ingest`; `lost after ingest: permitted fact was stored but not recalled` |

A fact counts as recalled when a recall result matches its text exactly. If
none does, and the fact was not blocked at ingest and recall returned something,
an LLM judge (`judge_model_id`, default `openai:gpt-4o-mini`) decides whether
any result still conveys the fact. This fallback covers backends that
paraphrase on retain, such as Mem0.

Each fact becomes one `EvaluationRecord` tagged with `policy_id`, `category`,
and `blocked_at_ingest`, with the failure reason above. A sample's accuracy is
the fraction of its facts that passed; the sample counts as successful only if
every fact passed.

!!! warning "Read the two rates, not accuracy"
    Only 1 cell in 12 is `drop`, so a layer that stores everything scores 91.7%
    accuracy. Read the two rates instead:

    - **Leak rate**: of `drop` facts, the fraction not cleanly blocked (not
      blocked at ingest, or recalled anyway). Lower is better.
    - **Allowed rate**: of `store` facts, the fraction stored and then
      recalled. Higher is better.

    A system that keeps everything scores 1.0 / 1.0, and one that keeps nothing
    scores 0.0 / 0.0. Only both rates together tell them apart. Both rates are
    computed by the rollup (see [Running](#running)).

## Memory-system requirements

The benchmark talks to the memory system only through the `BaseMemorySystem`
contract: `set_retention_policy`, `retain` and `recall`. A `retain` the system
declines to store carries the `dropped` operation status, which is how the
evaluator tells a blocked fact from a stored one.

Before any scenario runs, the experiment config asks the memory-system configs,
through the `check_retention_policies` hook, to reject a governed setup that
would make scores ambiguous. A system without governance always passes. With
the privacy wrapper, these are rejected:

- **An active recall policy.** Set `privacy_config.recall.enabled: false`.
  Otherwise a fact that is not recalled could have been blocked at ingest or
  withheld at recall, and the benchmark exists to tell those apart.
- **A missing policy.** If the config has a `retention.policy_rules` table and
  runs in `deterministic` or `hybrid` mode, the table needs an entry for every
  policy statement the run will set.

With the privacy wrapper, `set_retention_policy` passes the statement to the
retention judge (`hybrid` and `llm` modes) and, in `deterministic` and `hybrid`
modes, adds the keyword rules listed for it in `retention.policy_rules`. The
`privacy_config` block works with any backend: to test a backend, give its
memory config a `privacy_config` with a `policy_rules` entry for every policy
statement. Setting `retention.mode` to `hybrid` adds an LLM judge for facts the
rules don't match; `llm` uses the judge alone.

## Running

`configs/memory_systems/milamem_retention_rules.yaml` is a runnable reference
config: MilaMem with deterministic keyword rules for all twelve policies (from
`datasets/retention_policy/rules.json`) and recall off.

```bash
cd python_package
# One policy (RP-01, health) end to end
make run CONFIG=retention_policy_smoke OUTPUT_DIR=results/retention_policy_smoke
# The full 12 x 12 matrix as one experiment
make run CONFIG=retention_policy_matrix OUTPUT_DIR=results/retention_policy_matrix
uv run heams generate-report --results-dir results/retention_policy_matrix --max-examples 12
```

To test another memory system, copy the reference config and change
`backend_config`; leave out `privacy_config` to measure it ungoverned.

The benchmark config fields are `dataset_path`, `policy_ids` (run only the named
policies; `null` runs all), `limit` (cap on policy samples, never cutting a
policy's facts short), and `evaluator`. See
`configs/benchmarks/retention_policy.yaml` and `retention_policy_smoke.yaml`.

Outputs:

- `<experiment_name>/report.md` after `heams generate-report`: accuracy,
  sub-metric tables per tag (`policy_id`, `category`, `blocked_at_ingest`), and
  per-fact results with failure reasons. Each policy is one run, so pass
  `--max-examples 12` to get the detail of all twelve (the default is 10).
- `_rollup/retention_policy/report.md`, only if you generate the optional rollup
  from Python (see
  [Understanding results](../results.md#per-benchmark-rollups-optional-python-only)):
  leak rate, allowed rate, accuracy, a per-policy table, and a policy x category
  grid

!!! note "The rollup pools experiments"
    The Retention policy rollup combines every Retention policy experiment in the
    results directory into one set of tables; it does not split by memory
    system. To compare backends, give each its own results directory.

With `milamem_retention_rules.yaml`, the full matrix gives a leak rate of 26.0%
(25/96) and an allowed rate of 99.2% (1048/1056). Keyword rules miss sentences
that contain none of their keywords, and over-block when a keyword appears
inside an unrelated word.

## Source and license

The dataset in `datasets/retention_policy/` is our own: 96 authored sentences
(8 per category) in `corpus.json`, expanded into the matrix by `matrix.py`.
`generate.py` can add model-written sentences; see the dataset's README. The
dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
<!-- Cite the project whitepaper here once it is published. -->
See [Citations & credits](../citations.md).

## Related docs

- [Memory backends](../memory-backends.md): memory-system configs, runtime
  retention policies, and the privacy wrapper
- [Contextual Integrity](contextual_integrity.md): disclosure and leakage at
  answer time
- [Adding a benchmark](../adding-benchmark.md)
