# heams-memory-letta

[Letta](https://github.com/letta-ai/letta/tree/0.16.8) 0.16.8, the server that
implements [MemGPT](https://arxiv.org/abs/2310.08560), as a memory plugin for
HEAMS. Plugin type: `letta`. Preset:
`python_package/configs/memory_systems/campaigns/letta_openai.yaml`.

Each memory instance is one `memgpt_agent` with the paper's six memory tools.
Priming defaults to direct archival insertion; all other operations are messages to
the agent, which decides what to store and search for. See the "letta (external plugin)" section of
`python_package/docs/memory-backends.md` for how each operation maps onto MemGPT.

## Why 0.16.8

It is the last Python release of the Letta server. Unpinned, `letta` on PyPI is now
Letta Code (0.32+), a terminal app with no agents API. 0.16.8 is Postgres-only, and
the plugin hosts Letta's own app inside the benchmark process, so there is no Letta
server to start; the plugin starts the database itself.

## Setup

From the repository root:

```sh
make -C memory_plugins/heams_memory_letta setup
```

Then set the model provider's key in the repository `.env`:

```sh
OPENAI_API_KEY=...
```

That is all `heams run` needs, locally and on the cluster.

Letta 0.16.8 runs only on Postgres with pgvector, so each benchmark process starts a
private database and deletes it on exit; concurrent runs never share one.
PostgreSQL 17 and pgvector come from the [`pgembed`](https://github.com/Ladybug-Memory/pgembed)
wheel. Letta's schema comes from its Alembic migrations at the 0.16.8 tag, which the
pip package omits: `setup` downloads them once into
`~/.cache/heams-memory-letta/` (or `$XDG_CACHE_HOME`), checks the archive's SHA-256,
and starts a database to check that everything works. Starting a database adds
a few seconds to a run. If the migrations are missing, the first run downloads them.

To use an existing database instead, set `LETTA_PG_URI` to a Postgres with pgvector
and Letta's schema. Every memory instance creates its own agent and deletes it and
any plugin-created archive on close, so resources left in that database by an
interrupted run never affect a result.

## Tests

```sh
make -C memory_plugins/heams_memory_letta test-ci   # no database or API key needed
make -C memory_plugins/heams_memory_letta test-live # the campaign preset, one session + one recall
make -C memory_plugins/heams_memory_letta test      # both
```

`test-ci` covers both priming modes with a fake client. It makes no model calls and
starts no database. `test-live` needs `OPENAI_API_KEY`, starts a private database, and
uses the campaign preset's priming mode (insert by default); it is not needed for
unit validation.

## Things to know

- **Priming modes.** `prime_mode: insert` (default) writes each group directly into
  archival memory, preserving turn order, speaker labels, dates and epoch timestamps
  when supplied. It makes embedding calls, but no agent completions and adds no
  conversation messages. `prime_mode: agent` preserves MemGPT-managed ingestion:
  each group is a message and the agent chooses what to store in core or archival
  memory. `retain`, `recall` and `delete` always use the agent. These modes measure
  different ingestion behavior; distinguish them when comparing benchmark results.
- **Passage and completion limits.** `passage_max_tokens: 1024` caps insert-mode
  passages, including the repeated date header, using `cl100k_base` (the tokenizer
  for the presets' OpenAI embeddings). The maximum is 8191. Text is preserved across
  splits; a cap too small for the header and next character raises an error.
  `max_output_tokens: 4096` caps each agent completion, not the whole multi-step
  operation. The plugin also restores Letta's core-block character-limit checks and
  rejects `core_memory_replace` with empty `old_content`.
- **The private database.** Its files live under `SLURM_TMPDIR` when set (node-local
  on the cluster), otherwise the system temp directory, and Letta connects through a
  Unix socket under `/tmp`. Crash safety is off, since the database is discarded. It
  requires a password even locally, because compute nodes are shared. It is stopped
  and deleted on a normal exit, Ctrl-C, or `SIGTERM` (such as a Slurm time limit). A
  process killed with `SIGKILL` cannot stop it; Slurm normally ends a job's
  processes, and locally `pkill -f pgembed/pginstall/bin/postgres` does.
- **Dependency overrides.** Letta pins OpenTelemetry, `mcp`, `uvicorn`, `websockets`
  and `docstring-parser` to versions older than `python_package`'s; resolved as
  pinned they downgrade the benchmark agent's own dependencies. Both this package
  and `python_package` override them to `python_package`'s versions.
- **Several write tools need gpt-4o or gpt-4o-mini.** Letta 0.16.8 lets a request
  start with any one of several tools only for the models in its
  `STRUCTURED_OUTPUT_MODELS`. For any other model, `write_tools` must name a single
  tool.
- **`reasoning_effort` for gpt-5 models.** Letta 0.16.8 drops `reasoning_effort: none`
  from its Chat Completions requests, and OpenAI rejects a `gpt-5*` model's default
  effort alongside function tools, so the plugin puts it back.
- **Recall and delete leave no messages behind.** After each one the plugin puts the
  agent's context back and deletes the messages it added, so probes stay independent
  and a probe never becomes a memory. Recall cannot write; delete keeps its memory
  edits. A deployed MemGPT agent would keep these messages. The deletion calls
  Letta's message manager directly, since 0.16.8's REST API has no route for it.
- **Recall returns core memory and archival hits only.** Delete cannot reach the
  conversation log, so conversation hits could return a forgotten fact.
- **Log noise.** At startup Letta lists models from every provider it finds a key
  for, so 401 tracebacks for providers without a valid key are expected.
