# heams-memory-letta

[Letta](https://github.com/letta-ai/letta/tree/0.16.8) 0.16.8, the server that
implements [MemGPT](https://arxiv.org/abs/2310.08560), as a memory plugin for
HEAMS. Plugin type: `letta`. Preset:
`python_package/configs/memory_systems/campaigns/letta_openai.yaml`.

Each memory instance is one `memgpt_agent` with the paper's six memory tools, and
every memory operation is a message to that agent: it decides what to store and what
to search for. See the "letta (external plugin)" section of
`python_package/docs/memory-backends.md` for how each operation maps onto MemGPT.

## Why 0.16.8

It is the last Python release of the Letta server. Unpinned, `letta` on PyPI is now
Letta Code (0.32+), a terminal app with no agents API. 0.16.8 is Postgres-only, and
the plugin hosts Letta's own app inside the benchmark process, so there is a database
to set up but no Letta server to start.

## One-time setup

From the repository root, on macOS with Homebrew:

```sh
# 1. Postgres 17 + pgvector, running in the background
brew install postgresql@17 pgvector
brew services start postgresql@17
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"   # the formula is keg-only

# 2. the database Letta uses
createdb letta
psql -d letta -c 'create extension if not exists vector'

# 3. the plugin's environment (also installable with `uv sync --extra letta`
#    from python_package)
make -C memory_plugins/heams_memory_letta setup

# 4. Letta's schema, from the migrations at the same tag; the pip package omits them
REPO=$PWD
git clone --depth 1 -b 0.16.8 https://github.com/letta-ai/letta.git /tmp/letta-0.16.8
(cd /tmp/letta-0.16.8 && LETTA_PG_URI=postgresql://localhost:5432/letta \
    "$REPO/memory_plugins/heams_memory_letta/.venv/bin/alembic" upgrade head)
```

Check it: `psql -d letta -Atc 'select version_num from alembic_version'` prints
`1c28e167b74f`.

Then set the database and the model provider's key in the repository `.env`:

```sh
LETTA_PG_URI=postgresql://localhost:5432/letta
OPENAI_API_KEY=...
```

To start over, `dropdb letta` and repeat steps 2 and 4. Every memory instance creates
its own agent and deletes it on close, so agents left in the database by an
interrupted run never affect a result.

## Tests

```sh
make -C memory_plugins/heams_memory_letta test-ci   # no database or API key needed
make -C memory_plugins/heams_memory_letta test-live # the campaign preset, one session + one recall
make -C memory_plugins/heams_memory_letta test      # both
```

## Things to know

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
