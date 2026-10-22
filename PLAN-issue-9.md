# Plan: decouple the governance layer from the core framework

**Issue:** [milatechtransfer/heams#9](https://github.com/milatechtransfer/heams/issues/9)

## Goal

Governance (retention and recall policies) stops being part of the framework. The framework, the benchmarks and the evaluator interact with it only through the `BaseMemorySystem` contract. Any memory system can be governed, and every memory system, governed or not, can run the governance benchmarks.

**Done when:**

- `memory_systems/privacy` is gone from `python_package`, and the core code (`python_package/src`) no longer reads `privacy_config`.
- Orchestration, evaluation and benchmark code have no governance-specific logic beyond the `BaseMemorySystem` contract.
- MilaMem with governance runs the `retention_policy` benchmark, and so does any other memory system, governed or not.

## How it works today

The governance layer lives in `python_package/src/heams/memory_systems/privacy/`: 13 files, about 1,700 lines. It contains the config, the retention and recall policies in their deterministic, hybrid and LLM modes, and `PrivacyMemoryWrapper`, which wraps any backend.

The core depends on it in six places:

| File | Dependency |
|---|---|
| `memory_systems/factory.py` | `MemorySystemVariant.privacy_config`; `create_memory` wraps any backend in `PrivacyMemoryWrapper` |
| `orchestration/scenario_runtime.py` | Reads `privacy_config` from the variant and passes it to the factory |
| `orchestration/experiment_config.py` | Two Retention-policy validators that inspect `privacy_config`: recall must be disabled, and every benchmark policy must have `policy_rules` |
| `evaluation/scenarios/retention_policy.py` | Detects a dropped fact by comparing the response text to the wrapper's `RETENTION_DROPPED_MESSAGE` |
| `memory_systems/base.py` | The default `set_retention_policy` error tells users to configure `privacy_config` |
| `memory_systems/__init__.py` | Re-exports the privacy classes |

The Retention policy benchmark talks to the memory system through `MemoryManipulation` commands (`set_retention_policy`, `retain`, `recall`). `scenario_execution.py` turns each result into `AgentResponse(answer=str(outcome))`, and that text is all the evaluator sees.

Because the layer filters facts before the backend sees them, the benchmark today measures the governance layer rather than the memory system: the same layer on different backends makes identical drop decisions, and only recall of the kept facts differs.

## Design

The governance layer becomes the **privacy wrapper**: a plugin in `memory_plugins/` that wraps whichever memory system a config names, declared by the same top-level `privacy_config:` block as today.

- **No config change.** `backend_config` and `privacy_config:` stay exactly as they are. Only who reads `privacy_config` changes: the privacy wrapper plugin instead of the core.
- **Any memory system can be governed, with no work in its plugin.** Mem0, Letta, HippoRAG or MilaMem each run with the same governance layer by adding `privacy_config:` to an experiment's config.
- **The core stays generic.** It gains a small extension point for wrapper plugins, with nothing governance-specific in it.
- **The code already fits.** `PrivacyMemoryWrapper` already forwards retain, recall, delete, consolidate, priming, event sinks and run context to the backend it wraps, and presents the backend's config as its own (`self.config = inner.config`), so the orchestrator sees the same `top_k`, consolidation setting and storage as for the unwrapped system.
- **Native governance stays possible.** A memory system can also implement `set_retention_policy` itself, for example by adding the policy to its own prompts. The privacy wrapper is one implementation of the contract, not the only one, and the benchmark can compare the two.

| Question | Decision |
|---|---|
| Where the governance layer lives | The privacy wrapper plugin, `memory_plugins/heams_memory_privacy_wrapper/` |
| How it is configured | The existing top-level `privacy_config:` block, unchanged |
| How a dropped `retain` is signalled | A structured operation status, not a shared message string |
| The checks that fail a run before it starts | Kept for governed systems, through a generic contract hook (see below) |
| A memory system without governance | Runs the benchmark and gets an ungoverned score |
| A runnable Retention policy config | Shipped: MilaMem with `privacy_config`, plus a run config |

### Governed and ungoverned memory systems

Every memory system can run the Retention policy benchmark:

- **Governed** (wrapped by the privacy wrapper, or implementing `set_retention_policy` natively): it enforces the policy each sample sets, and is scored on what it drops and what it keeps.
- **Ungoverned** (neither): the default `set_retention_policy` records that no policy is enforced, the system stores everything, and it gets an ungoverned score, the same role `no_memory` plays on the performance benchmarks.

The pre-run checks only apply to governed systems. They reject a governed system that is **misconfigured** for the benchmark, for example with a recall policy that would hide facts, or a rule-based mode with no rules for one of the benchmark's policies. Their default does nothing, so an ungoverned system always passes and runs.

## Config shape

Unchanged:

```yaml
id: milamem_retention_rules
backend_config:          # the memory system
  type: milamem
  ...
privacy_config:          # now read by the privacy wrapper plugin
  retention: {...}
  recall: {...}
  rules: [...]
```

## Changes

### 1. Add a wrapper-plugin extension point to the core

- A `MemoryWrapperPlugin` descriptor, next to `MemoryPlugin` in `memory_systems/plugins.py`: the top-level config key it claims, its config model, and a `wrap(memory, config) -> BaseMemorySystem` function.
- Discovery through a new entry point group, `heams.memory_wrappers`, alongside `heams.memory_systems`.
- `MemorySystemVariant` (`factory.py`) accepts top-level keys claimed by installed wrapper plugins and validates each with that plugin's config model. A key that no installed plugin claims is still an error, with a message naming the plugin to install for `privacy_config`. Serialization writes the blocks back under their keys, so saved experiment configs keep the same shape.
- `create_memory` builds the backend as today, then applies each declared wrapper.
- `scenario_runtime.py` passes the variant's wrapper configs instead of `privacy_config`.

### 2. Create the privacy wrapper plugin

- New package `memory_plugins/heams_memory_privacy_wrapper/` (distribution `heams-memory-privacy-wrapper`), laid out like the other plugins: `pyproject.toml`, `Makefile`, `README.md`, `src/`, `tests/`, registered under `heams.memory_wrappers` with the key `privacy_config`.
- Its `README.md` opens by stating that it is a wrapper, not a memory system: it adds retention and recall policies around any memory system.
- Move `python_package/src/heams/memory_systems/privacy/` into it, keeping retention and recall behavior identical. `PrivacyMemoryWrapper` becomes the plugin's wrap function.
- Add the plugin to `python_package`'s `all-memory-plugins` extra and the lock files.

### 3. Signal a dropped `retain` with a structured status

- Add `DROPPED = "dropped"` to `MemoryOperationStatus` in `memory_systems/schemas.py`.
- In the wrapper, a dropped `retain` logs a `retain` event with status `dropped`. Today it returns the message without logging anything.
- In `scenario_execution.py`, record the status of the memory event a `MemoryManipulation` produced on the `AgentResponse`, as a new optional field.
- `evaluation/scenarios/retention_policy.py` checks that status and no longer imports anything from the governance code.

### 4. Keep the pre-run checks for governed systems through a contract hook

- Add one generic method, for example `check_retention_policies(policies: list[str]) -> None`, to the backend config base class (`BaseMemoryConfig`) and to the wrapper config base class. It raises `ValueError` when a governed system is misconfigured for the given policies. The default does nothing, so ungoverned systems pass.
- The checks run on configs, before any memory system exists, as today.
- `orchestration/experiment_config.py` calls it on the backend config and on each wrapper config before a `retention_policy` run, instead of inspecting `privacy_config`.
- The privacy wrapper's config implements today's two checks: recall must be disabled, and in rule-based modes every policy needs `policy_rules`.

### 5. Remove governance from the core

- `factory.py`: remove the `privacy_config` field and the privacy-specific wrapping; the key is now handled by the extension point.
- `memory_systems/__init__.py`: remove the privacy exports.
- `base.py`: the default `set_retention_policy` no longer raises. It returns that no policy is enforced and logs the call, so the run's results show the system was ungoverned.
- The Retention policy report states, per experiment, whether a policy was enforced, so an ungoverned score isn't read as a governed one.

### 6. Configs

- No changes to existing configs: `milamem_retention_and_recall_{rules,hybrid,llm}.yaml` and `configs/runs/canonical_governance_ablation.yaml` keep working as they are.
- Ship a Retention policy config, MilaMem with `privacy_config` using deterministic keyword rules from `datasets/retention_policy/rules.json` and recall disabled, plus a run config, so the benchmark runs out of the box.

### 7. Tests

- Core tests for the extension point: a test wrapper plugin is discovered, its key is validated, an unclaimed key is rejected, configs round-trip through serialization, and `create_memory` applies the wrapper.
- Move `tests/memory_systems/privacy/test_privacy_contracts.py` and `test_privacy_integration.py` into the privacy wrapper's tests, and add a test wrapping at least two different memory systems.
- Core Retention-benchmark tests (`tests/benchmarks/retention_policy/`: `fixtures.py`, `test_benchmark.py`, `test_evaluator.py`) switch from the privacy wrapper to a small fake memory system that implements the contract: `set_retention_policy`, the `dropped` status and the check hook. This shows that any memory system can run the benchmark without the privacy wrapper.
- A test that an ungoverned memory system runs the benchmark and is reported as ungoverned.
- Update the other tests that use `privacy_config`: `test_memory_factory.py`, `test_experiment_config.py`, `test_orchestrator.py` and `test_scenario_runtime.py`.
- Add a test that runs the Retention policy benchmark on MilaMem with `privacy_config`.

### 8. Docs

- The privacy wrapper's `README.md`: what a wrapper is, the `privacy_config` block, and the modes.
- `docs/memory-backends.md`: move the `privacy_config` section under the privacy wrapper plugin.
- `docs/adding-memory-system.md`: how to write a wrapper plugin, and the two ways a memory system supports governance (the privacy wrapper, or implementing `set_retention_policy` natively), the `dropped` status, and the check hook.
- `docs/benchmarks/retention_policy.md` and `datasets/retention_policy/README.md`: governed and ungoverned runs, the new config, and how the benchmark uses the contract.
- `docs/architecture.md`, `docs/getting-started.md`, `docs/cli.md` and the root `README.md`: describe `privacy_config` as provided by the privacy wrapper plugin, and list the plugin.

## Alternatives considered

**Native governance only, with the privacy layer as a library.** The privacy code would become a regular package that memory-system plugins import and call from their own `retain`, `recall` and `set_retention_policy`, with no wrapper plugin and no extension point in the core.

- **For:** a simpler core, and one way to do governance instead of two.
- **Against:**
  - Only memory systems whose plugin integrates the library can be governed. Today any memory system can be, so Mem0, Letta and HippoRAG would lose governance until each plugin is changed.
  - Users who want governance would depend on plugin code changes instead of adding a config block.
  - Privacy settings would move into each plugin's `backend_config`, a config change for every governed setup, and their shape could differ between plugins.
  - With per-plugin integration, two memory systems could apply the same policy differently, making governed comparisons less direct.

Native governance stays available with the privacy wrapper, so a plugin can still integrate governance itself later.

## Open questions

- The order of several wrappers, should more than one ever be declared on the same config. Not needed for the privacy wrapper alone; config order is the simplest rule.

## Out of scope

- Native `set_retention_policy` for Mem0 (the policy added to its fact-extraction prompt) and Letta (the policy given to its agent as an instruction).
- An agent-mediated variant of the Retention policy benchmark, where facts arrive as user messages and the agent, told the policy, decides what to store. Today the benchmark writes facts with `MemoryManipulation` commands that go straight to the memory system, so the agent never sees them.

## Size

Large: probably 50–60 files, including a new plugin package. Most of it is moving code and tests and updating references. The new logic is the wrapper extension point (1), the `dropped` status (3) and the contract hook (4).

## Verification

- `make test-ci`, `make docs-strict` and `make check` pass, including the new plugin's tests.
- No `memory_systems.privacy`, `privacy_config` or `RETENTION_DROPPED_MESSAGE` remains in `python_package/src`.
- The existing governed configs load and run unchanged, and `canonical_governance_ablation.yaml` still resolves.
- The shipped Retention config runs the benchmark end to end. To confirm nothing changed, run the same settings on MilaMem with the current core wrapper before the change, then with the plugin after, and compare the leak and allowed rates.
- The privacy wrapper on a second memory system (for example `in_context`) also runs the benchmark, and an ungoverned memory system runs it and is reported as ungoverned.

## Risks

- **Plugin not installed:** a config with `privacy_config` fails to load when the privacy wrapper plugin isn't installed. The error must name the plugin to install. It is part of `all-memory-plugins`, which `make setup` installs.
- **Behavior drift:** moving the wrapper must not change retention or recall decisions. The before-and-after rates in Verification check this.
- **Telling governed and ungoverned runs apart:** with the backend config unchanged, a governed and an ungoverned MilaMem run share the memory type `milamem`. Reports label runs by the config's `id`, falling back to the memory type (`reporting/aggregation.py`), and the governed configs each set their own `id`. A governed config without an `id` would show only `milamem`, so the docs should say to set one.
