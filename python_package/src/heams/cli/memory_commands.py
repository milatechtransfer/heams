from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from heams.memory_systems.api import BaseMemorySystem, MemoryOperationStatus
from heams.memory_systems.plugins import (
    MemoryPlugin,
    MemoryRuntimeContext,
    get_memory_plugin_catalog,
)


def list_memory_plugins() -> int:
    for plugin in get_memory_plugin_catalog().plugins():
        print(plugin.type_name)
    return 0


def test_memory_plugins() -> int:
    has_failures = False
    for plugin in get_memory_plugin_catalog().plugins():
        status, detail = _test_memory_plugin(plugin)
        if status == "failure":
            has_failures = True
        suffix = f" ({detail})" if detail else ""
        print(f"{plugin.type_name}: {status}{suffix}")
    return 1 if has_failures else 0


def _test_memory_plugin(plugin: MemoryPlugin) -> tuple[str, str | None]:
    try:
        config = plugin.config_model.model_validate({})
    except ValidationError:
        return "setup-required", "configuration is required"

    with TemporaryDirectory(prefix="heams-memory-test-") as temporary_dir:
        workspace_dir = Path(temporary_dir)
        events = []
        runtime = MemoryRuntimeContext(
            storage_dir=workspace_dir,
            event_sink=events,
            run_context={"diagnostic": "heams memory test"},
        )
        memory: BaseMemorySystem | None = None
        failure: Exception | None = None
        retain_result: object = None
        recall_result: object = None
        try:
            created_memory = plugin.create(config, runtime)
            if not isinstance(created_memory, BaseMemorySystem):
                failure = TypeError("factory did not return BaseMemorySystem")
            else:
                memory = created_memory
                retain_result = memory.retain(
                    "heams memory diagnostic sentinel",
                    {"source": "heams memory test"},
                )
                recall_result = memory.recall(
                    "heams memory diagnostic sentinel",
                    top_k=1,
                )
        except Exception as exc:
            failure = exc
        finally:
            if memory is not None:
                try:
                    memory.close()
                except Exception as exc:
                    if failure is None:
                        failure = exc

        if failure is not None:
            return "failure", str(failure)

    if any(
        event.status == MemoryOperationStatus.ERROR or event.error for event in events
    ):
        return "failure", "memory operation emitted an error"
    if not isinstance(retain_result, str):
        return "failure", "retain returned a non-string result"
    if not isinstance(recall_result, list) or not all(
        isinstance(item, str) for item in recall_result
    ):
        return "failure", "recall returned a non-string list"
    if any(event.status == MemoryOperationStatus.SKIPPED for event in events):
        return "pass", "operation skipped by backend"
    return "pass", None
