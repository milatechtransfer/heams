# Erasability & Editability (EE) benchmark

The EE benchmark measures two things a long-term memory system is supposed to do when a
conversation invalidates what it already stored:

| Task | Name | Passes when the response… |
|---|---|---|
| `erase` | **Erasability** (explicit deletion) | treats a fact the user asked it to forget as gone |
| `edit` | **Editability** (cascade update) | reports the *new* value of an entity that depended on a changed root fact |

## Dataset

The EE benchmark is built on
[`meme-benchmark/MEME`](https://huggingface.co/datasets/meme-benchmark/MEME)
(CC-BY-4.0), using the **`filler32k`** config: `meme_filler32k.json`, ~20 MB, 100
episodes of ~33k–37k tokens each.

Every episode plants a dependency chain of facts across 17–27 sessions, then fires
exactly one **Change+Delete Event** session that both

1. changes a *root* entity (`lactose intolerance` → `high blood pressure`), and
2. explicitly asks the assistant to forget one fact
   (`"My partner is James. Please remove that from your memory."`).

Probes are asked twice — once before that session, once after it. The two blocks
ask the *same questions*; only the gold answers change.

MEME's raw tags are translated at the dataset boundary — `Del` → `erase`, `Cas` → `edit` —
so everything downstream (config, probe keys, report tags) reads in this benchmark's own
vocabulary. MEME also ships `Abs`, `Tr`, `Agg` and `ER` task types. Those are **out of scope**
and are dropped at the dataset boundary, leaving **264 before-probes and 264
after-probes** (1 `erase` + 1–2 `edit` per episode).

## Scenario shape

Each episode becomes **one scenario** with three phases:

```
scenario_priming.memory  <- every session EXCEPT the Change+Delete session
execution_sequence:
  1. before-probes    2-3 x UserStimulus   (phase: "before")
  2. change replay    4   x UserStimulus   (ungraded ingestion turns)
  3. after-probes     2-3 x UserStimulus   (phase: "after")
```

The priming rule is one line: *everything except the change+delete session*.
Priming runs through the real `retain()` pipeline, so the haystack is ingested the
way the backend would ingest it normally.

By default (`agent_conversation: stateful`) every turn sees the agent's earlier
messages, so after-probes can read the replayed change from the conversation
itself. Set `agent_conversation: independent` to reset the agent's messages
before each turn; after-probes then learn the change only from memory.

## Evaluator

Grading follows the reference MEME implementation
([`SeokwonJung-Jay/MEME-public`](https://github.com/SeokwonJung-Jay/MEME-public),
`code/eval/judge.py`).

**One LLM call per probe.** Each call sees only its own question, gold value and
answer. Batching probes into one call would let the before-answer leak into the
after-verdict, which is exactly the comparison the benchmark is measuring.

**Three rubrics**, all authored in
`configs/benchmarks/full_campaign/erasability_editability.yaml` with no Python
fallbacks. Each
takes `{question}`, `{gold}`, `{agent_answer}` and returns
`{"correct": bool, "reason": str}`:

| Field | Ported from | Applies to |
|---|---|---|
| `before_prompt` | `BEFORE_COMMON_PROMPT` | every before-probe, both task types |
| `deletion_prompt` | `DELETION_PROMPT` | `erase` after-probes |
| `cascade_prompt` | `CASCADE_D_PROMPT` | `edit` after-probes |

The after-phase rubrics are strict: a cascade answer fails
if it lists *both* old and new values, hedges, or uses future tense ("will change
to X"); a deletion answer fails if it reveals the deleted value **in any way, even
while stating that it was deleted**.

## Metrics

The optional EE rollup, `_rollup/ee/report.md` (generated from Python; see
[Understanding results](../results.md#per-benchmark-rollups-optional-python-only)),
leads with the **trivial-pass filtered score**, then shows the raw judge pass rate
as a diagnostic.

### Why filtering is necessary

For an `erase` probe the correct after-answer is "I don't have that." A memory system
that **never learned the fact** answers the same way — and scores a pass. Run a
no-memory baseline on episode `pl_001`, entity `partner`:

| phase | question | gold | baseline answer | raw verdict |
|---|---|---|---|---|
| before | *What's my partner's name?* | `James` | "I don't know" | ✗ |
| after | *What's my partner's name?* | `James` (must not appear) | "I don't know" | **✓** |

Unfiltered, a system that stored nothing scores 100% on deletion.

### Per-task scores

The rollup leads with one row per memory backend and one column per task type:

| Memory backend | edit_accuracy | erase_accuracy |
| --- | ---: | ---: |
| basic_complete_memory_system | 16.7% (1/6) | 60.0% (3/5) |
| no_memory | 0.0% (0/6) | 0.0% (0/5) |

Both apply the same trivial-pass rule as the headline — `real` over every graded
after-probe of that task type — so a system cannot inflate `erase_accuracy` with
facts it never learned. A task type absent from an episode is omitted rather than
reported as a misleading `0`.

The same numbers appear in `audit_notes` inside `runs.jsonl`:

```
Score 4/11 = 36% - counts only 'real' (...). Buckets: real=4, trivial=2,
knew_but_failed=5, never_knew=0. Per task: edit_accuracy=0.167, erase_accuracy=0.600.
```

### Probe keys

Each graded probe gets a key `{episode_id}:{task_type}:{phase}{n}`, where `n` is
1-based and restarts per task and phase — so the key set shows the shape of an
episode at a glance:

```
pl_001:edit:before1   pl_001:edit:before2   pl_001:erase:before1
pl_001:edit:after1    pl_001:edit:after2    pl_001:erase:after1
```

That episode has two edit probes and one erase probe, in each phase.

### The four buckets

Crossing the two phases **per entity** separates the cases:

| before | after | bucket | meaning | counts? |
|---|---|---|---|---|
| ✓ | ✓ | `real` | learned it, then correctly forgot/updated it | **yes** |
| ✗ | ✓ | `trivial` | never learned it, so the after-answer was correct for free | no |
| ✓ | ✗ | `knew_but_failed` | learned it, but failed to forget/update it | no |
| ✗ | ✗ | `never_knew` | never learned it, and answered wrong after too | no |

These wordings come from `EE_PASS_TYPE_LABELS` in `judge.py`, which is the single
source of truth: the bucket order, the `pass_type_label` tag on every record, the
`audit_notes` line and the rollup legend all read from it, so they cannot drift
apart. Each graded after-probe carries both `pass_type` and `pass_type_label`, so
a record in `runs.jsonl` is interpretable without consulting the docs.

The published dataset card defines the score as *"correct only if the system also
answered the corresponding `before_questions` correctly"* — that is
**`real / total`**, and it is what `accuracy_score` and the headline table report.
Only `real` counts; the other three buckets are shown beside it so an inflated
deletion number is visible rather than hidden.

The raw stratified table below it reports ungated per-probe verdicts by task type,
phase, cascade hop (1 or 2) and domain (`personal_life` / `software_project`). It
does **not** apply the filter and will read higher than the headline; treat it as
a diagnostic.

## Running benchmark

From `python_package/`:

```sh
# First MEME episode (downloads meme_filler32k.json on first use).
uv run heams run --config configs/runs/testing/erasability_editability_smoke.yaml \
                --output-dir results/erasability_editability_smoke

# Full 100-episode run (downloads meme_filler32k.json on first use).
uv run heams run --config configs/runs/no_memory_all_benchmarks/erasability_editability.yaml \
                --output-dir results/erasability_editability
```

## Source and license

MEME's paper is under anonymous review, so cite the
[`meme-benchmark/MEME`](https://huggingface.co/datasets/meme-benchmark/MEME)
dataset (**CC BY 4.0**); the reference
[code](https://github.com/SeokwonJung-Jay/MEME-public) is **MIT**. The dataset is
downloaded on first use; only a shortened copy of one episode is committed, as a
test fixture.
See [Citations & credits](../citations.md).
