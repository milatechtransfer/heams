"""Run the front-end unit tests (`frontend/*.test.mjs`) with Node's runner.

They cover the browser's pure modules (escaping, formatting, routing,
filters). Skipped when Node.js is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND_TESTS = Path(__file__).resolve().parent / "frontend"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_frontend_unit_tests_pass() -> None:
    tests = sorted(str(path) for path in FRONTEND_TESTS.glob("*.test.mjs"))
    assert tests, f"no front-end tests found in {FRONTEND_TESTS}"

    result = subprocess.run(
        ["node", "--test", *tests],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
