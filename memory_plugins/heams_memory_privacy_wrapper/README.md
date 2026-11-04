# heams-memory-privacy-wrapper

The privacy wrapper for HEAMS. **It is a wrapper, not a memory system:** it adds
retention and recall policies around whichever memory system a config names, so
any memory system can be governed without changes to its own plugin.

A memory-system config enables it with a top-level `privacy_config` block next to
`backend_config`:

```yaml
id: milamem_retention_rules
backend_config:          # the memory system being wrapped
  type: milamem
  ...
privacy_config:          # read by this plugin
  retention:
    mode: deterministic
    policy_rules:
      "Do not retain the user's health information.":
        - rule_id: drop-health
          content_substrings: [health, medical, doctor]
  recall:
    enabled: false
```

Set an `id` on wrapped configs: reports label runs by `id`, so a wrapped and an
unwrapped run of the same memory system stay distinguishable.

## What it does

- **Retention** decides, before content reaches the memory system, whether to
  store or drop it, in `deterministic`, `hybrid` or `llm` mode. A dropped
  `retain` is logged with the `dropped` operation status.
- **Recall** filters what the memory system returns, in the same three modes.
- **`set_retention_policy`** sets the user's free-text policies at runtime; the
  Retention policy benchmark uses it to apply each sample's policy.

The modes, the shared `rules` ruleset, `policy_rules`, and the judge prompts are
documented in the "Privacy wrapper" and "Runtime retention policies" sections of
`python_package/docs/memory-backends.md`.

## How it plugs in

The plugin registers under the `heams.memory_wrappers` entry point group with
the config key `privacy_config`. The framework validates that block with
`PrivacyConfig`, builds the memory system from `backend_config`, and passes it
to `wrap_memory_with_privacy`. An inactive `privacy_config` leaves the memory
system unwrapped.

Before a Retention policy run, the framework calls
`PrivacyConfig.check_retention_policies`, which rejects an active recall policy
and a `policy_rules` table missing one of the benchmark's policies.

## Install and test

`make setup` at the repository root installs this plugin with the others
(the `all-memory-plugins` extra). To work on it alone:

```sh
make -C memory_plugins/heams_memory_privacy_wrapper setup
make -C memory_plugins/heams_memory_privacy_wrapper test-ci
```
