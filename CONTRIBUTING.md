# Contributing

Thank you for your interest in HEAMS. The framework runs today, and the
contributions that help it most are bug reports, bug fixes, and targeted
improvements to what is already here. New benchmarks and memory systems are
welcome too, but they are larger pieces of work, so start them with an issue.

## What we can and cannot promise

We would rather tell you this plainly than have you work it out from our
silence.

- **We read every issue and pull request. We do not reply to all of them.**
  A small team cannot hold a conversation with everybody who turns up. There
  is no response time, and nothing here is a commitment to review or merge a
  particular change.
- **An issue going quiet is not a verdict.** It usually means the week got
  away from us. Comment on it again.
- **Problems that several people hit move first.** If an existing issue
  describes what you see, add your details there rather than opening a new
  one.
- **You do not need us to extend HEAMS.** A memory system is a plugin package
  that can live in its own repository, and you can run your own benchmark
  configs from a fork. Sending it upstream is welcome, not required.

## Reporting bugs and asking questions

Search the [existing issues](https://github.com/milatechtransfer/heams/issues)
first, then open a new one. For a bug, include:

- the command you ran and the run config it used
- the commit you are on (`git rev-parse HEAD`)
- the agent model, provider, and memory system, if the bug depends on them
- the full error output

If the problem is a wrong score rather than an error, say which benchmark and
which samples, what the framework produced, and what you expected instead.

Do not attach API keys, `.env` files, or execution traces from private data:
traces contain the prompts and recalled memories of a run.

## Where a change goes

| The change | Where it goes |
| --- | --- |
| A bug fix, with a test that reproduces the bug | A pull request |
| A bug you cannot fix yourself | An issue |
| A documentation fix | A pull request |
| A small, self-contained improvement (an error message, a missing option, a performance fix) | A pull request |
| A change to how an existing benchmark is prompted, sampled, or scored | An issue first |
| A change to the run config format, the plugin API, or orchestration | An issue first |
| A new benchmark | An issue first, then a pull request |
| A new memory system | A plugin in its own repository; an issue if you want it shipped here |
| A question or a research collaboration | An issue |

"An issue first" means describing the change and why you want it, then
waiting for a maintainer to respond before you do the work. This is the
cheapest moment to find out we had already decided something.

## Changes that move benchmark scores

Published results, from the HEAMS paper and from anyone who has reported
numbers with the framework, depend on how each benchmark is prompted, sampled,
and scored. A change to any of these changes the benchmark, even when it fixes
a bug. In the pull request:

- say that it changes scores, and for which benchmarks
- show results before and after the change on the same run config, with the
  agent model and memory system you used
- update the benchmark's page in `python_package/docs/benchmarks/`

## Setting up

Follow [Setup](python_package/docs/getting-started.md#setup) in the
documentation, then install the pre-commit hook so formatting and lint checks
run on every commit:

```sh
make setup-pre-commit
```

## Adding a benchmark or a memory system

Open an issue first for either one (see [Where a change goes](#where-a-change-goes)).

- **Memory system:** follow
  [Adding a memory system](python_package/docs/adding-memory-system.md). A memory
  system is a plugin package, so it can live in its own repository without
  changes to the framework.
- **Benchmark:** follow
  [Adding a benchmark](python_package/docs/adding-benchmark.md). If you work with
  a coding agent, point it at
  [`.agents/skills/implement-benchmark/`](.agents/skills/implement-benchmark/).
  The dataset must be downloaded from its original source rather than committed,
  and listed with its paper and license in
  [Citations & credits](python_package/docs/citations.md).

## Pull requests

1. Fork the repository and branch from `main`. Branches on this repository are
   for maintainers.
2. Keep each pull request to one change. If you cannot describe it in one
   sentence, it is probably two.
3. Add or update tests for the behavior you change. For a bug fix, add a test
   that fails without the fix.
4. Update the pages in `python_package/docs/` that describe it; the docs are
   the user-facing reference and should match the code.
5. Run the same checks as CI, from the repository root:

   ```sh
   make check        # formatting and lint (pre-commit)
   make test-ci      # tests, excluding integration and Ollama tests
   make docs-strict  # docs build, failing on warnings
   ```

   If you change a memory plugin or a model provider, also run `make test`,
   which adds the integration and Ollama tests. The Ollama tests need a local
   Ollama server with `qwen2.5:7b` pulled, and skip themselves when no server
   is running.

6. In the pull request, describe what changed and why, say how you checked it
   (which tests, which run config, which model), and link the issue it
   addresses.
7. Keep the title readable. Pull requests are squash merged, so the title is
   what survives in the history.

## Security

Do not open a public issue for a security problem, such as a way to leak API
keys or trace contents, or to make a run execute untrusted code. Report it
privately through the repository's
[Security tab](https://github.com/milatechtransfer/heams/security) using
"Report a vulnerability".

## License

By contributing, you agree that your contributions are released under the
project's Apache License 2.0 (and, for datasets, under CC BY 4.0), as described
in the [README](README.md#license). Apache 2.0 covers this in section 5, so
there is no contributor license agreement to sign.

## Research collaborations

If your group would like to use the framework in a study, evaluate a memory
system with it, or contribute a benchmark, open an issue describing what you
have in mind. We read every one, but we cannot take on every collaboration.
