# Plan: decouple the governance layer from the core framework

**Issue:** [milatechtransfer/heams#9](https://github.com/milatechtransfer/heams/issues/9)

## Goal

Governance (retention and recall policies) stops being part of the framework. The framework, the benchmarks and the evaluator interact with it only through the `BaseMemorySystem` contract. Any memory system can be governed, and any memory system that implements the contract can run the governance benchmarks.

**Done when:**

- `memory_systems/privacy` and `privacy_config` are gone from `python_package`.
- Orchestration, evaluation and benchmark code have no governance-specific logic beyond the `BaseMemorySystem` contract.
- MilaMem with governance runs the `retention_policy` benchmark, and so can any other memory system that implements the contract.

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

Governance becomes a **wrapper plugin**: a plugin that wraps whichever memory system a config names, declared by a top-level block next to `backend_config`, as `privacy_config` is today.

- **Minimal config change.** `backend_config` is untouched; the top-level `privacy_config:` key becomes `governance:`, with the same contents.
- **Any memory system can be governed.** Mem0, Letta, HippoRAG or MilaMem can each run with the same governance layer.
- **The core stays generic.** It gains a small extension point for wrapper plugins, with nothing governance-specific in it. Other wrappers could use the same mechanism later.
- **The code already fits.** `PrivacyMemoryWrapper` already forwards retain, recall, delete, consolidate, priming, event sinks and run context to the backend it wraps, and presents the backend's config as its own (`self.config = inner.config`), so the orchestrator sees the same `top_k`, consolidation setting and storage as for the unwrapped system.
- **Native governance stays possible.** A memory system can also implement `set_retention_policy` itself, for example by adding the policy to its own prompts. The governance plugin is one implementation of the contract, not the only one, and the benchmark can compare the two.

| Question | Decision |
|---|---|
| Where governance lives | A new wrapper plugin, `memory_plugins/heams_memory_governance/` |
| How it is configured | A top-level `governance:` block next to `backend_config`, with the same contents as today's `privacy_config:` |
| How a dropped `retain` is signalled | A structured operation status, not a shared message string |
| The checks that fail a run before it starts | Kept, through a generic contract hook |
| A memory system without governance | Can still run the benchmark: the default `set_retention_policy` records that no policy is enforced, so the system gets an ungoverned score |
| A runnable Retention policy config | Shipped: MilaMem with governance, plus a run config |

## Config shape

Today:

```yaml
id: milamem_retention_rules
backend_config:
  type: milamem
  ...
privacy_config:
  retention: {...}
  recall: {...}
  rules: [...]
```

With the wrapper plugin:

```yaml
id: milamem_retention_rules
backend_config:          # unchanged
  type: milamem
  ...
governance:              # renamed from privacy_config; same contents
  retention: {...}
  recall: {...}
  rules: [...]
```

## Changes

### 1. Add a wrapper-plugin extension point to the core

- A `MemoryWrapperPlugin` descriptor, next to `MemoryPlugin` in `memory_systems/plugins.py`: the top-level config key it claims, its config model, and a `wrap(memory, config) -> BaseMemorySystem` function.
- Discovery through a new entry point group, `heams.memory_wrappers`, alongside `heams.memory_systems`.
- `MemorySystemVariant` (`factory.py`) accepts top-level keys claimed by installed wrapper plugins and validates each with that plugin's config model. A key that no installed plugin claims is still an error, and serialization writes the blocks back under their keys, so saved experiment configs keep the same shape.
- `create_memory` builds the backend as today, then applies each declared wrapper.
- `scenario_runtime.py` passes the variant's wrapper configs instead of `privacy_config`.

### 2. Create the governance plugin

- New package `memory_plugins/heams_memory_governance/` (distribution `heams-memory-governance`), laid out like the other plugins: `pyproject.toml`, `Makefile`, `README.md`, `src/`, `tests/`, registered under `heams.memory_wrappers` with the key `governance`.
- Move `python_package/src/heams/memory_systems/privacy/` into it, keeping retention and recall behavior identical. `PrivacyMemoryWrapper` becomes the plugin's wrap function.
- Add the plugin to `python_package`'s `all-memory-plugins` extra and the lock files.

### 3. Signal a dropped `retain` with a structured status

- Add `DROPPED = "dropped"` to `MemoryOperationStatus` in `memory_systems/schemas.py`.
- In the wrapper, a dropped `retain` logs a `retain` event with status `dropped`. Today it returns the message without logging anything.
- In `scenario_execution.py`, record the status of the memory event a `MemoryManipulation` produced on the `AgentResponse`, as a new optional field.
- `evaluation/scenarios/retention_policy.py` checks that status and no longer imports anything from the governance code.

### 4. Keep the fail-before-run checks through a contract hook

- Add one generic method, for example `check_retention_policies(policies: list[str]) -> None`, to the backend config base class (`BaseMemoryConfig`) and to the wrapper config base class. It raises `ValueError` when the memory system can't enforce the given policies under the benchmark's conditions. The default does nothing, so systems without governance pass.
- The checks run on configs, before any memory system exists, as today.
- `orchestration/experiment_config.py` calls it on the backend config and on each wrapper config before a `retention_policy` run, instead of inspecting `privacy_config`.
- The governance plugin's config implements today's two checks: recall must be disabled, and in rule-based modes every policy needs `policy_rules`.

### 5. Remove `privacy_config` from the core

- `factory.py`: remove the `privacy_config` field and the privacy-specific wrapping. A config that still has a top-level `privacy_config` fails with a clear error pointing to `governance:`.
- `memory_systems/__init__.py`: remove the privacy exports.
- `base.py`: the default `set_retention_policy` no longer raises. It returns that no policy is enforced and logs the call, so the run's results show the system was ungoverned. Every memory system can then run the benchmark and get an honest ungoverned score, the same role `no_memory` plays on the performance benchmarks.
- The Retention policy report states, per experiment, whether a policy was enforced, so an ungoverned score isn't read as a governed one.

### 6. Configs

- In `milamem_retention_and_recall_{rules,hybrid,llm}.yaml`, rename `privacy_config:` to `governance:`. `configs/runs/canonical_governance_ablation.yaml` uses them, so its paths stay the same.
- Ship a Retention policy config, MilaMem with `governance:` using deterministic keyword rules from `datasets/retention_policy/rules.json` and recall disabled, plus a run config, so the benchmark runs out of the box.

### 7. Tests

- Core tests for the extension point: a test wrapper plugin is discovered, its key is validated, an unclaimed key is rejected, configs round-trip through serialization, and `create_memory` applies the wrapper.
- Move `tests/memory_systems/privacy/test_privacy_contracts.py` and `test_privacy_integration.py` into the governance plugin's tests, and add a test wrapping at least two different memory systems.
- Core Retention-benchmark tests (`tests/benchmarks/retention_policy/`: `fixtures.py`, `test_benchmark.py`, `test_evaluator.py`) switch from the privacy wrapper to a small fake memory system that implements the contract: `set_retention_policy`, the `dropped` status and the check hook. This shows that any memory system can run the benchmark without the governance plugin.
- Update the other tests that use `privacy_config`: `test_memory_factory.py`, `test_experiment_config.py`, `test_orchestrator.py` and `test_scenario_runtime.py`.
- Add a test that runs the Retention policy benchmark on MilaMem with governance.

### 8. Docs

- The governance plugin's `README.md`: the `governance:` block, the modes, and how wrapping works.
- `docs/memory-backends.md`: replace the `privacy_config` section with the governance plugin, and update the examples.
- `docs/adding-memory-system.md`: how to write a wrapper plugin, and the two ways a memory system supports governance (the governance plugin, or implementing `set_retention_policy` natively), the `dropped` status, and the check hook.
- `docs/benchmarks/retention_policy.md` and `datasets/retention_policy/README.md`: the new config, and how the benchmark uses the contract.
- `docs/architecture.md`, `docs/getting-started.md`, `docs/cli.md` and the root `README.md`: replace `privacy_config` references and list the governance plugin.

## Open questions

- The key name: `governance` is proposed. Keeping `privacy_config` would mean no config change at all, but leaves that name in `python_package`'s configs.
- The order of several wrappers, should more than one ever be declared on the same config. Not needed for governance alone; config order is the simplest rule.

## Out of scope

- Native `set_retention_policy` for Mem0 (the policy added to its fact-extraction prompt) and Letta (the policy given to its agent as an instruction).
- An agent-mediated variant of the Retention policy benchmark, where facts arrive as user messages and the agent, told the policy, decides what to store. Today the benchmark writes facts with `MemoryManipulation` commands that go straight to the memory system, so the agent never sees them.

## Size

Large: probably 50–60 files, including a new plugin package. Most of it is moving code and tests and updating references. The new logic is the wrapper extension point (1), the `dropped` status (3) and the contract hook (4).

## Verification

- `make test-ci`, `make docs-strict` and `make check` pass, including the new plugin's tests.
- No `privacy_config`, `memory_systems.privacy` or `RETENTION_DROPPED_MESSAGE` remains in `python_package/src`.
- The shipped Retention config runs the benchmark end to end. To confirm nothing changed, run the same governance settings on MilaMem with the current core wrapper before the change, then with the plugin after, and compare the leak and allowed rates.
- Governance on a second memory system (for example `in_context`) also runs the benchmark.
- `canonical_governance_ablation.yaml` still resolves with the migrated configs.

## Risks

- **Config breakage:** any config outside this repository that uses top-level `privacy_config` stops loading. The error message must point to `governance:`.
- **Behavior drift:** moving the wrapper must not change retention or recall decisions. The before-and-after rates in Verification check this.
- **Telling governed and ungoverned runs apart:** with the backend config unchanged, a governed and an ungoverned MilaMem run share the memory type `milamem`. Reports label runs by the config's `id`, falling back to the memory type (`reporting/aggregation.py`), and the governance configs each set their own `id`. A governed config without an `id` would show only `milamem`, so the docs should say to set one.
