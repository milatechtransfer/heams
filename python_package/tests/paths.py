"""Stable roots so nested tests do not count Path(__file__).parents."""

from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TESTS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
