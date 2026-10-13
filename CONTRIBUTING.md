# Contributing

Thank you for your interest in HEAMS. Bug reports, new
benchmarks, new memory systems, and documentation fixes are all welcome.

## Reporting bugs and asking questions

Open an [issue on GitHub](https://github.com/milatechtransfer/heams/issues).
For a bug, include:

- the command you ran and the run config it used
- the commit you are on (`git rev-parse HEAD`)
- the full error output

Do not attach API keys, `.env` files, or execution traces from private data:
traces contain the prompts and recalled memories of a run.

## Setting up

Follow [Install](README.md#install) in the README, then install the pre-commit
hook so formatting and lint checks run on every commit:

```sh
make setup-pre-commit
```

## Adding a benchmark or a memory system

- **Memory system:** follow
  [Adding a memory system](python_package/docs/adding-memory-system.md). A memory
  system is a plugin package, so it can live in its own repository without
  changes to the framework.
- **Benchmark:** follow
  [Adding a benchmark](python_package/docs/adding-benchmark.md). If you work with
  a coding agent, point it at
  [`.agents/skills/implement-benchmark/`](.agents/skills/implement-benchmark/).

## Pull requests

1. Create a branch from `main` and keep each pull request to one change.
2. Add or update tests for the behavior you change.
3. Update the pages in `python_package/docs/` that describe it; the docs are
   the user-facing reference and should match the code.
4. Run the same checks as CI, from the repository root:

   ```sh
   make check        # formatting and lint (pre-commit)
   make test-ci      # tests, excluding integration and Ollama tests
   make docs-strict  # docs build, failing on warnings
   ```

5. Describe what changed and why in the pull request, and link the issue it
   addresses.

## License

By contributing, you agree that your contributions are released under the
project's MIT License (and, for datasets, under CC BY 4.0), as described in the
[README](README.md#license).

## Research collaborations

If your group would like to use the framework in a study, evaluate a memory
system with it, or contribute a benchmark, open an issue describing what you
have in mind and the maintainers will follow up.
