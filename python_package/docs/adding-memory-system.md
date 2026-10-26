# Adding a memory system

A memory system is added as a **plugin**: a separate Python package that the
framework discovers when it is installed in the same environment. You do not
edit the core package. The working reference is the example plugin in
`memory_plugins/heams_memory_example/`.

How the pieces fit:

1. Your package defines a config class, a memory-system class and a
   `MemoryPlugin` descriptor.
2. Its `pyproject.toml` exposes the descriptor through the `heams.memory_systems`
   entry point, so `heams` finds it once the package is installed.
3. A YAML file under `configs/memory_systems/` selects it by its `type`.
4. A run config pairs that YAML with an agent and a benchmark, and `heams run`
   evaluates it.

## 1. Import from the public API only

Import everything from `heams.memory_systems.api`. Other modules under
`heams.memory_systems` are internal and may change.

```python
from heams.memory_systems.api import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryEntry,
    MemoryEventTarget,
    MemoryOperationResult,
    MemoryPlugin,
    MemoryRuntimeContext,
    render_tool_memory_entry,
)
```

## 2. Define the config

Subclass `BaseMemoryConfig` and give it a `type` literal. The `type` is how
YAML files select your system, so make it unique and namespaced (for example
`acme.graph_memory`, not `graph_memory`).

```python
from typing import Literal

from pydantic import Field


class AcmeMemoryConfig(BaseMemoryConfig):
    type: Literal["acme.memory"] = "acme.memory"
    max_entries: int = Field(default=100, gt=0)
```

Every config inherits these fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `type` | required | Selects the plugin |
| `name` | `agent_memory` | Display name used in telemetry |
| `top_k` | `5` | Default number of results for `recall` |
| `consolidation.enabled` | `false` | When true, the framework calls `consolidate()` after priming and again when the sample finishes |

Unknown fields are rejected, so typos in YAML fail early. If your backend keeps
files on disk, override `with_runtime_storage(storage_dir)` to return a copy of
the config that points your storage at the per-sample `storage_dir` (the
HippoRAG, mem0 and milamem plugins do this).

## 3. Implement the memory system

Subclass `BaseMemorySystem`. Keep the keyword-only constructor arguments and
pass them to `super().__init__`; the framework uses them for storage and
event logging.

```python
from pathlib import Path
from typing import Any


class AcmeMemorySystem(BaseMemorySystem):
    config: AcmeMemoryConfig

    def __init__(
        self,
        config: AcmeMemoryConfig,
        *,
        storage_dir: Path,
        event_sink: MemoryEventTarget | None = None,
        run_context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            config,
            storage_dir=storage_dir,
            event_sink=event_sink,
            run_context=run_context,
        )
        self._entries: list[MemoryEntry] = []

    def _retain_impl(
        self, content: str, metadata: dict[str, Any]
    ) -> MemoryOperationResult:
        self._entries.append(MemoryEntry(content=content, metadata=metadata))
        return MemoryOperationResult(
            value="Saved to memory.",
            result={"write_count": 1},
        )

    def _recall_impl(self, query: str, top_k: int) -> MemoryOperationResult:
        matches = [e for e in self._entries if query.casefold() in e.content.casefold()]
        items = [render_tool_memory_entry(e) for e in matches[-top_k:]]
        return MemoryOperationResult(
            value=items,
            result={
                "returned_items": items,
                "result_count": len(items),
                "output_bytes": self._memory_size_proxy(items),
            },
        )
```

### Methods you implement

| Method | Required | `value` to return |
|--------|----------|-------------------|
| `_retain_impl(content, metadata)` | Yes | A short confirmation string |
| `_recall_impl(query, top_k)` | Yes | A `list[str]`; this is what the agent sees |
| `_delete_impl(query)` | No | A summary string. The default reports the operation as `skipped` |
| `_consolidate_impl(context, metadata)` | No | A summary string. The default reports the operation as `skipped` |
| `prime_memory_entries(entries)` | No | Nothing. Override it to load a benchmark's priming data in bulk; the default calls `retain()` for each entry |
| `close()` | No | Nothing. Release connections, files or servers |

Do **not** replace the public `retain`, `recall`, `delete` and `consolidate`
methods. They are what the agent's memory tools and the benchmark call, and the
base class wraps your `_*_impl` methods to time them, log an event for each
call, and turn an exception into an `error` event plus a failure message for
the agent. If you need to wrap one, for example to hold a lock as the Letta
plugin does, call `super()` so the timing and logging still happen.

You don't have to implement `set_retention_policy`, which the Retention policy
benchmark uses. Without it, your system runs that benchmark ungoverned: the
default enforces no policy and says so in the results. There are two ways to
give it governance:

- **Wrap it with the privacy wrapper.** Add a `privacy_config` block next to
  `backend_config` in the memory-system YAML; the privacy wrapper plugin then
  applies retention and recall policies around your system, with no code
  changes. See [Privacy wrapper](memory-backends.md#privacy-wrapper).
- **Implement it natively.** Override `set_retention_policy(policies)` to enforce
  the free-text policies your own way, for example by adding them to your
  system's prompts. When `retain` declines to store something, log the `retain`
  event with the `dropped` status, so benchmarks see the content was blocked.
  To reject a configuration that can't enforce a benchmark's policies before
  the run starts, override `check_retention_policies(policies)` on your config
  class and raise `ValueError`.

### `MemoryOperationResult`

| Field | Meaning |
|-------|---------|
| `value` | What the caller receives (see the table above) |
| `status` | `"success"` (default), `"skipped"` (the operation is not supported) or `"error"` |
| `result` | Facts about the outcome, recorded in the event and read by the metrics (see below) |
| `payload` | Extra facts about the request, merged into the event's payload |
| `error` | An error message, when relevant |

### `result` keys

Some keys are read by the framework's metrics and trace viewer; the rest are
conventions that keep plugins consistent.

| Operation | Key | Used for |
|-----------|-----|----------|
| recall | `returned_items` | Retrieval-quality metrics (hit@k, MRR); give the same list as `value` |
| recall | `result_count` (or `returned_count`) | Efficiency metrics: results per recall |
| recall | `output_bytes` (or `output_size_chars`) | Efficiency metrics: recall output size |
| any | `summary` or `message` | Text shown for the event in execution traces |
| retain | `write_count` | Convention: number of items written |
| delete | `deleted_count` | Convention: number of items removed |
| consolidate | `changed_count` | Convention: number of items merged or rewritten |
| any | `supported` | Convention: `False` when the operation is not available |

The request side (content size, query, `top_k`, run context) is recorded by the
base class; you do not need to add it.

### Helpers

| Helper | Use it to |
|--------|-----------|
| `render_tool_memory_entry(entry)` | Format a `MemoryEntry` (content plus metadata) as a recall item |
| `self._memory_size_proxy(value)` | Measure a value consistently for `output_bytes` |
| `self._emit_memory_operation(...)` | Log a retain event yourself when `prime_memory_entries` loads data without calling `retain()` (as the HippoRAG, mem0 and Letta plugins do) |

## 4. Expose the plugin

Create a factory and a `MemoryPlugin` descriptor in your package:

```python
def _create_memory(
    config: BaseMemoryConfig, runtime: MemoryRuntimeContext
) -> BaseMemorySystem:
    return AcmeMemorySystem(
        config=config,
        storage_dir=runtime.storage_dir,
        event_sink=runtime.event_sink,
        run_context=runtime.run_context,
    )


plugin = MemoryPlugin(
    type_name="acme.memory",
    config_model=AcmeMemoryConfig,
    create=_create_memory,
)
```

The framework calls `create` once per sample, with a fresh `storage_dir`, so
state never leaks between samples. `type_name` must match the config's `type`
and be unique among installed plugins; a duplicate stops discovery with an
error.

The contract is synchronous: each operation returns when it is done. Entry
points load and run installed Python code; they are not a security sandbox, so
only install plugins you trust.

Declare the entry point in your package's `pyproject.toml`:

```toml
[project.entry-points."heams.memory_systems"]
acme_memory = "acme_memory.plugin:plugin"
```

Then install the package into the framework's environment, for example:

```sh
cd python_package
uv pip install -e ../path/to/acme_memory
```

`make setup` (and `uv sync`) reset the environment to the lock file, so
reinstall your package afterwards. Plugins bundled in this repository are wired
as extras in `python_package/pyproject.toml` instead.

## 5. Add a config file

Create a YAML under `configs/memory_systems/`. `backend_config` is validated
against your config class:

```yaml
id: acme_memory
backend_config:
  type: acme.memory
  name: acme.memory
  top_k: 5
  max_entries: 100
```

Reference it from a run config's `memory_system_config_file`, as the configs
under `configs/runs/` do.

## 6. Check it

```sh
cd python_package
uv run heams memory list
uv run heams memory test
```

`memory list` shows every discovered plugin type. `memory test` creates each
plugin in a temporary directory, retains a sentinel and recalls it, and reports
`pass`, `failure` or `setup-required`. `setup-required` means the plugin's
config cannot be built without settings (for example a model or an API key);
test such plugins with a real run config instead. `memory test` does not
exercise delete or consolidate, so cover those in your own tests.

To run your plugin on a benchmark, copy
`configs/runs/examples/compare_memory_systems.yaml`, point one experiment's
`memory_system_config_file` at your YAML, and run it as described in
[Running experiments](getting-started.md#running-experiments).

For a complete, working version of steps 1 to 6, including a real
`_delete_impl` and `_consolidate_impl`, read
`memory_plugins/heams_memory_example/src/heams_memory_example/plugin.py`.

## 7. Wrapping an existing memory system

Most plugins wrap a library or service that already stores and searches
memories. The bundled plugins show the patterns below; each row names the one to
read. HippoRAG (`memory_plugins/heams_memory_hipporag/`) is the smallest and the
best place to start; mem0 (`memory_plugins/heams_memory_mem0/`) adds more
configuration; Letta (`memory_plugins/heams_memory_letta/`) wraps a running
service.

| Pattern | What to do | Shown by |
|---------|------------|----------|
| Library settings in the config | Pass the library's own settings through unchanged, or as typed sub-configs translated in one place | HippoRAG passes `global_config` as is; mem0 translates `llm`, `embedder` and `vector_store` in `to_mem0_config()` |
| All state under `storage_dir` | Override `with_runtime_storage` so every file, database and collection lives in the per-sample folder | HippoRAG sets `save_dir`; mem0 sets the vector store path and a per-sample collection name, and its constructor puts the history database in `storage_dir` |
| Create in `__init__`, release in `close()` | Build the client in the constructor and release everything it holds in `close()` | mem0 closes each underlying client; Letta deletes its agent and only logs a warning if that fails |
| Map retain and recall onto the library | `_retain_impl` calls the library's add or index and reports an honest `write_count`. `_recall_impl` calls its search and formats each hit with `render_tool_memory_entry`, keeping the library's metadata. Extra `result` keys are fine | HippoRAG counts stored passages before and after indexing, formats hits with `render_tool_memory_entry`, and adds `scores` and `graph_seeds`; mem0 builds the same JSON itself and adds `scores` |
| Convert metadata the library cannot store | Translate metadata into the types the store accepts, and translate it back on recall | mem0 (`_to_mem0_metadata` and `_rehydrate_mem0_metadata`, for Chroma) |
| Delete and consolidate only when native | Use the library's own operation and report `skipped` when it cannot act; otherwise keep the defaults | HippoRAG uses its native delete and reports `skipped` when nothing matches; mem0 keeps both defaults |
| Bulk priming | Override `prime_memory_entries` to load a whole conversation at once, using the `ingest_*` hints below, and log one retain event per group with `_emit_memory_operation` | HippoRAG, mem0 and Letta |
| Wrapping a service | Share one client per process, fail at start-up if a setting or the service version is wrong, keep secrets out of the config by naming an environment variable, remove per-sample remote resources in `close()`, and wrap public methods (calling `super()`) when calls must not overlap | Letta (Postgres URI from the variable named in `pg_uri_env`, version check, one agent per sample, a lock) |

### Ingestion hints

Benchmarks that prime memory with a conversation (LoCoMo, LongMemEval and BEAM)
add `ingest_*` keys to each `MemoryEntry`'s metadata, so a plugin can rebuild
the conversation instead of storing turns one by one:

| Key | Meaning |
|-----|---------|
| `ingest_group_id` | Turns that belong together, such as one session |
| `ingest_order` | Position of the turn within its group |
| `ingest_role` | The speaker's role, normally `user` or `assistant` |
| `ingest_speaker_name` | The speaker's name, when the benchmark has one (LoCoMo) |
| `ingest_timestamp_epoch` | When the turn happened, as a Unix timestamp |
| `ingest_scope_id` | The sample the turn belongs to |

They are hints for ingestion, not facts about the memory: HippoRAG drops them
from the metadata of the passages it builds while priming. Entries without
them should be stored one by one, which is what the default
`prime_memory_entries` does.
