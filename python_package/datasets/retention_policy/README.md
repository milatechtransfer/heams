# Retention Policy dataset

A 12 x 12 matrix. Twelve categories of atomic sentences, twelve prohibition-only
policies, every sentence evaluated under every policy.

```
corpus.json      12 categories x 8 authored sentences
policies.json    12 policies, one per category
rules.json       keyword rules for the deterministic baseline arm
matrix.py        builds dataset.json from the three above
generate.py      adds model-written sentences to corpus.json
dataset.json     generated -- what the benchmark reads
```

## The matrix

Policies are prohibition-only and each names exactly one category, so the label
is one comparison:

```python
expected = "drop" if sentence.category == policy.prohibits else "store"
```

`drop` on the diagonal, `store` everywhere else:

```
                creden educat financ govern health  legal locati    pet prefer relati travel   work
RP-01 health         .      .      .      .   DROP      .      .      .      .      .      .      .
RP-02 work           .      .      .      .      .      .      .      .      .      .      .   DROP
RP-03 pet            .      .      .      .      .      .      .   DROP      .      .      .      .
...
```

Nothing about the text changes between cells -- only the policy does. **Every
sentence carries both labels** depending on which policy is active, which is the
property the benchmark exists to test: a layer that has memorised which topics
look sensitive cannot pass, because the same sentence must be kept under eleven
policies and dropped under the twelfth.

## Building it

```bash
python datasets/retention_policy/matrix.py                  # 4/category -> 576 rows
python datasets/retention_policy/matrix.py --per-category 8 # 8/category -> 1152 rows
python datasets/retention_policy/matrix.py --render         # print the grid
```

`matrix.py` refuses to build if the corpus is impure -- a sentence that strays
into another category would be mislabelled in more than one cell, since every
sentence is judged twelve times.

## Growing the corpus

`generate.py` asks a model for more sentences in a category and appends the ones
that pass vetting. It never touches existing entries.

```bash
python datasets/retention_policy/generate.py --category legal --count 6
python datasets/retention_policy/generate.py --all --count 4
python datasets/retention_policy/generate.py --category pet --count 4 --dry-run
python datasets/retention_policy/matrix.py --per-category 8   # then use them
```

The model is told which category to write for; it never sees a policy and never
assigns a label, so it cannot influence what the right answer is. Candidates are
rejected for length, multiple sentences, first person, joining two facts with
"and", duplicates, and -- via the same `strays()` the build uses -- for touching
another category. Sharing that one definition means the generator cannot accept
something `matrix.py` would later reject.

Two things to check after a batch:

- **Read what it wrote.** The vetting is mechanical: it confirms a sentence is
  well-formed and category-pure, not that it is a sensible thing for a person to
  have said.
- **Re-run the baseline.** Purity is checked against `matrix.py`'s cue lists,
  which are not the same words as `rules.json`. A new sentence can be correctly
  categorised and still trip another policy's keyword rules -- "at home" fires
  the location ruleset, for instance. That shows up as off-diagonal
  over-blocking, and it is the number that drifts as the corpus grows.

## Running it

Each benchmark sample is one policy. It opens by handing the memory system that
policy's statement (`set_retention_policy`), so the policy in force always
matches the rows being scored. The whole matrix runs as **one experiment**
against one memory-system config.

A keyword layer can't read a statement, so a deterministic memory config
carries `retention.policy_rules`, mapping every statement in `policies.json` to
its keyword list from `rules.json`; `configs/memory_systems/milamem_retention_rules.yaml`
does this for all twelve policies. The `privacy_config` block is backend-agnostic,
since the privacy wrapper plugin wraps any memory system, built-in or plugin. To
evaluate a backend, give it a memory config with that `privacy_config` block, and
point a run config's `memory_system_config_file` at it. Without the block, the
backend runs ungoverned.

## Read the two rates, not accuracy

Only 1 cell in 12 is `drop`, so **a layer that stores everything scores 91.7%
accuracy**. `matrix.py` prints this warning on every build.

- **leak rate** -- of `drop` rows, the fraction not cleanly blocked: stored at ingest or recalled anyway (diagonal, lower better)
- **allowed rate** -- of `store` rows, the fraction stored and then recalled (off-diagonal, higher better)

The off-diagonal is what makes over-blocking visible: a layer that drops
credentials under a health-only policy fails cells no single-policy run would
have exercised.

## Keyword rules

The rules in `rules.json` are derived from each **policy statement**, not from
the corpus -- deriving them from the sentences would let the baseline score
itself. An earlier corpus-derived ruleset produced a flattering 2.1% leak; treat
that kind of number with suspicion.

Keyword rules fail two ways, both instructive:

- **Diagonal misses** -- keyword brittleness. `"Feeds Juniper twice daily using a
  measured scoop."` contains no pet keyword.
- **Off-diagonal over-blocking** -- `"cat"` is a substring of `"medication"`, so
  the pet policy wrongly drops health sentences. The engine matches with a plain
  `substring in content`, which has caused this exact class of bug three times in
  this project.

This measures keyword coverage, not comprehension. Switching a memory config to
`mode: hybrid` or `mode: llm` runs the same dataset against a judge instead --
no benchmark changes needed, since the benchmark only reads the verdict.

## License

The data files (`corpus.json`, `policies.json`, `rules.json`, `dataset.json`)
are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The scripts that build them are part of the framework's code, under the
repository's [MIT License](../../../LICENSE).
