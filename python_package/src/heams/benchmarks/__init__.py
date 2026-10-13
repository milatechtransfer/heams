"""Benchmark package.

Kept empty so importing a judge submodule does not load ``benchmarks.base``.
That module imports ``scenarios``, which imports ``evaluation.factory`` while
the factory is still loading the judge.
"""
